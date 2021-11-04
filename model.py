from transformers import BertPreTrainedModel
import torch.nn.functional as F
from csrl_bert import CSRLBert
from model_utils import *


class CSAGN(BertPreTrainedModel):
    def __init__(self, config, wp, wf, n_speakers=2, mode="max", g_dim=100, num_layers=4, intra_loss=False,
                 inter_loss=False, inter_relation_num=4):
        super().__init__(config)

        self.config = config
        self.bert = CSRLBert(config)

        self.wp = wp
        self.wf = wf
        self.mode = mode
        self.g_dim = g_dim

        self.intra_loss = intra_loss
        self.inter_loss = inter_loss
        self.inter_relation_num = inter_relation_num

        self.edge_att = EdgeAtt(input_dim=config.hidden_size, wp=wp, wf=wf)
        self.gcn = GraphNet(config.hidden_size, g_dim, n_speakers)

        self.pred_aware_att = TencentSelfAtt(hidden_size=config.hidden_size, num_hidden_layers=num_layers, in_config=config)

        self.token_embedding_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.token_fusion_layer = MHA(config)

        self.speaker_embedding_proj = nn.Linear(config.hidden_size + g_dim, config.hidden_size, bias=True)
        self.speaker_fusion_layer = MHA(config)

        edge_type_to_idx = {}
        for j in range(1, n_speakers + 1):
            for k in range(1, n_speakers + 1):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx

        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.inter_classifier = nn.Linear(config.hidden_size + g_dim, inter_relation_num)
        self.intra_proj = nn.Linear(4 * config.hidden_size, 2 * config.hidden_size)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            text_lens=None,
            position_ids=None,
            pred_ids=None,
            head_mask=None,
            inputs_embeds=None,
            speaker_ids=None,
            labels=None,
            cls_vec=None,
            utt_labels=None,
            utt_mask=None,
            last_label=None,
            turn_ids=None
    ):
        context_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pred_ids=pred_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            turn_ids=turn_ids
        )  # (bsz, token_num, dim)

        bsz, seq_len, _ = context_features.shape
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1).expand((bsz, 1, seq_len, seq_len))
        attention_mask = (1 - attention_mask) * -10000.0

        # token level
        last_utt_embed = self.pred_aware_att(context_features, attention_mask=utt_mask)[0]
        y = self.token_embedding_proj(torch.cat([context_features, last_utt_embed], dim=-1))
        token_level_feat = self.token_fusion_layer(y, y, attention_mask)[0]

        # utterance level
        utt_features, cls_list = extract_utt_feat(token_level_feat, cls_vec, mode=self.mode)  # (bsz, turn_num, dim)
        utt_node_feat, utt_edge_idx, utt_edge_norm, utt_edge_type, edge_index_lengths = batch_graphify(
            utt_features, speaker_ids, text_lens, self.wp, self.wf, self.edge_type_to_idx, self.edge_att,
            self.device
        )  # (turn_num, g_dim)
        utt_graph_out = self.gcn(utt_node_feat, utt_edge_idx, utt_edge_norm, utt_edge_type)  # (turn_num, g_dim)
        # (bsz, seq_len, hidden_size)
        speaker_gcn_embed = flatten_graph_out(utt_graph_out, cls_list, context_feat=context_features,
                                              text_len=text_lens)  # (bsz, seq_len, g_dim)
        y = self.speaker_embedding_proj(torch.cat([token_level_feat, speaker_gcn_embed], dim=-1))
        utt_level_feat = self.speaker_fusion_layer(y, y, attention_mask)[0]

        logits = self.classifier(torch.cat([token_level_feat, utt_level_feat], dim=-1))  # (bsz, token_num, num_label)

        prob = F.softmax(logits, dim=-1)
        _, prediction = torch.max(prob, dim=-1)

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs.view(-1, self.config.num_labels), labels.view(-1), reduction='mean',
                              ignore_index=-100)
            if self.inter_loss:
                utt_labels = utt_labels[utt_labels > 0]
                inter_log_probs = F.log_softmax(
                    self.inter_classifier(torch.cat([utt_node_feat, utt_graph_out], dim=-1)), dim=-1)
                inter_loss = F.nll_loss(inter_log_probs.view(-1, self.inter_relation_num), utt_labels.view(-1),
                                        reduction='mean', ignore_index=0)
                # print("inter_loss: {}".format(inter_loss))
                loss += inter_loss
            if self.intra_loss:
                intra_log_probs = F.log_softmax(self.classifier(self.intra_proj(torch.cat(
                    [last_utt_embed, last_utt_embed - token_level_feat, last_utt_embed * token_level_feat,
                     token_level_feat], dim=-1))), dim=-1)
                intra_loss = F.nll_loss(intra_log_probs.view(-1, self.config.num_labels), last_label.view(-1),
                                        reduction='mean', ignore_index=-100)
                # print("intra_loss: {}".format(intra_loss))
                loss += intra_loss
            return loss, prediction

        return logits, prediction
