import copy

from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
import torch.nn as nn
import torch


class CSRLEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.pred_embedding = nn.Embedding(3, config.hidden_size, padding_idx=0)
        self.turn_embedding = nn.Embedding(11, config.hidden_size, padding_idx=0)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, pred_ids=None,
                turn_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        turn_embeddings = self.turn_embedding(turn_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        pred_embeddings = self.pred_embedding(pred_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + pred_embeddings + turn_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CSRLBert(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.csrl_embeddings = CSRLEmbeddings(config)

    def clone_embeddings(self):
        self.csrl_embeddings.word_embeddings = copy.deepcopy(self.embeddings.word_embeddings)
        self.csrl_embeddings.position_embeddings = copy.deepcopy(self.embeddings.position_embeddings)
        self.csrl_embeddings.token_type_embeddings = copy.deepcopy(self.embeddings.token_type_embeddings)
        self.csrl_embeddings.LayerNorm = copy.deepcopy(self.embeddings.LayerNorm)
        self.csrl_embeddings.dropout = copy.deepcopy(self.embeddings.dropout)
        self._init_weights(self.csrl_embeddings.pred_embedding)
        self._init_weights(self.csrl_embeddings.turn_embedding)

    def resize_seg_type_embeddings(self, new_num_seg_types):
        old_embeddings = self.embeddings.token_type_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_seg_types, use_old_embeddings=False)
        self.embeddings.token_type_embeddings = new_embeddings
        return self.embeddings.token_type_embeddings

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, pred_ids=None,
                turn_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.csrl_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            pred_ids=pred_ids, turn_ids=turn_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs[0]  # sequence_output, pooled_output, (hidden_states), (attentions)

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None, use_old_embeddings=True):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end
        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        if use_old_embeddings:
            # Copy word embeddings from the previous weights
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings
