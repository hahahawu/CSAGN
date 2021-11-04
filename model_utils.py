import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np
import math
import copy


class EdgeAtt(nn.Module):
    def __init__(self, input_dim, wp, wf):
        super(EdgeAtt, self).__init__()
        self.wp = wp
        self.wf = wf

        self.weight = nn.Parameter(torch.zeros((input_dim, input_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, text_len_tensor, edge_idn):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        att_matrix = torch.matmul(weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = text_len_tensor[i]
            alpha = torch.zeros((mx_len, 110)).to(node_features.device)
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat)
                probs = F.softmax(score, dim=0)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)

        return torch.stack(alphas, dim=0)


class GraphNet(nn.Module):

    def __init__(self, input_size, hidden_size, n_speakers=2):
        super(GraphNet, self).__init__()
        self.num_relations = 2 * n_speakers ** 2
        self.conv1 = RGCNConv(input_size, hidden_size, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # for using edge_norm, the version of torch-geometric should be downgrade to 1.4.3
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)

        return x


class TencentSelfAtt(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, in_config=None):
        super(TencentSelfAtt, self).__init__()
        if in_config is None:
            config = BertConfig()
        else:
            config = copy.deepcopy(in_config)
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        self.config = config
        self.encoder = BertEncoder(config)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(self, x, attention_mask=None, head_mask=None):
        device = x.device
        input_shape = x.size()[:-1]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        outputs = self.encoder(x, attention_mask=extended_attention_mask, head_mask=head_mask)
        return outputs


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # (batch_size, num_attention_heads, seq_len, attention_head_size) -> (batch_size, seq_len, num_attention_heads,
        # attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
                .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
                .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim=-1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim=-1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2


def batch_graphify(features, speaker_tensor, lengths, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()

            if item[0] == length_sum - 1:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + str(c)])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def extract_utt_feat(features, cls_position, mode='max'):
    bsz, turn_num = cls_position.shape
    _, seq_len, dim = features.shape
    utt_feat = torch.zeros((bsz, turn_num, dim)).to(features.device)

    utt_list = []

    for i in range(bsz):
        feat = features[i]  # (token_num, dim)
        cur_cls = cls_position[i][cls_position[i] > 0].tolist() + [seq_len - 1]  # (turn_num)
        utt_list.append(cur_cls)
        for x in range(1, len(cur_cls)):
            s = cur_cls[x-1]
            e = cur_cls[x]
            tmp_utt_embed = feat[s:e, :]    # (L', D)
            if mode == "mean":
                tmp_utt_embed = torch.mean(tmp_utt_embed, dim=0)   # (D)
            elif mode == 'sum':
                tmp_utt_embed = torch.sum(tmp_utt_embed, dim=0)
            elif mode == 'max':
                tmp_utt_embed = torch.max(tmp_utt_embed, dim=0)[0]
            else:
                raise ValueError("mode must be sum, mean or max, not {}".format(mode))
            utt_feat[i, x-1, :] = tmp_utt_embed
    return utt_feat, utt_list


def flatten_graph_out(graph_out, cls_list, context_feat=None, text_len=None):
    total_turn_num, dim = graph_out.shape
    bsz, seq_len, _ = context_feat.shape
    assert total_turn_num == sum([len(l) - 1 for l in cls_list])
    speaker_gcn_embed = torch.zeros(bsz, seq_len, dim).to(graph_out.device)

    cnt = 0
    for i in range(bsz):
        cl = cls_list[i]
        cur_turn_num = text_len[i]
        speaker_gcn_embed[i, 0, :] = torch.mean(graph_out[cnt:cnt+cur_turn_num, :], dim=0)
        for x in range(1, len(cl)):
            s = cl[x-1]
            e = cl[x]
            speaker_gcn_embed[i, s:e, :] = graph_out[cnt, :]
            cnt += 1
    return speaker_gcn_embed
