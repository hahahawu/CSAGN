from torch.utils import data
from dataset_utils import load_file, bert_vectorize_data
import torch


class SRLSet(data.Dataset):
    def __init__(self, data_path, bert_tokenizer, label_vocab, mode, seg_type_id_map, max_size=-1, use_seg=False):
        if type(data_path) == list:
            srl_data = []
            for p in data_path:
                srl_data.extend(load_file(p))
        else:
            srl_data = load_file(data_path)

        if max_size != -1:
            srl_data = srl_data[:max_size]

        self.instance = srl_data
        self.mode = mode
        self.use_seg = use_seg
        self.bert_tokenizer = bert_tokenizer
        self.label_vocab = label_vocab
        self.seg_type_id_map = seg_type_id_map

        print("there are ... {} cases for {}".format(len(self.instance), mode))

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        sen_vec, segment_ids, pred_vec, input_mask, label_vec, speaker_vec, turn_vec, cls_pos, utt_labels, utt_mask,\
        last_turn_label = bert_vectorize_data(self.bert_tokenizer, self.instance[index], self.label_vocab,
                                              self.seg_type_id_map, self.mode, self.use_seg)

        return InputExample(tokens=sen_vec, pred_vec=pred_vec, speaker_vec=speaker_vec, seg_vec=segment_ids,
                            label=label_vec, input_mask=input_mask, cls_position=cls_pos, utt_labels=utt_labels,
                            utt_mask=utt_mask, last_turn_label=last_turn_label, turn_vec=turn_vec)


class InputExample(object):
    def __init__(self, tokens, pred_vec, speaker_vec, turn_vec, seg_vec, label, input_mask, cls_position, utt_mask,
                 utt_labels=None, word_lens=None, last_turn_label=None):
        self.tokens = tokens
        self.pred_vec = pred_vec
        self.speaker_vec = speaker_vec
        self.seg_vec = seg_vec
        self.label = label
        self.word_lens = word_lens
        self.input_mask = input_mask
        self.cls_position = cls_position
        self.utt_labels = utt_labels
        self.utt_mask = utt_mask
        self.last_turn_label = last_turn_label
        self.turn_vec = turn_vec


def collate_fn(batch):
    sorted(batch, key=lambda x: len(x.tokens),  reverse=True)
    max_tokens = max([len(b.tokens) for b in batch])
    max_turns = max([len(b.speaker_vec) for b in batch])

    pad_tokens, pad_pred, pad_seg, pad_speaker, pad_turn, pad_label, pad_mask, pad_cls, pad_utt_labels, pad_utt_mask, pad_last_label = [], [], [], [], [], [], [], [], [], [], []

    for i in range(len(batch)):
        feat: InputExample = batch[i]
        tmp_wv = [0 for _ in range(max_tokens)]
        tmp_pred = [0 for _ in range(max_tokens)]
        tmp_seg = [0 for _ in range(max_tokens)]
        tmp_speaker = [0 for _ in range(max_turns)]
        tmp_label = [-100 for _ in range(max_tokens)]
        tmp_mask = [0 for _ in range(max_tokens)]
        tmp_cls = [-100 for _ in range(max_turns)]
        tmp_u_la = [0 for _ in range(max_turns)]
        tmp_utt_mask = [[0 for _ in range(max_tokens)] for _ in range(max_tokens)]
        tmp_last_label = [-100 for _ in range(max_tokens)]
        tmp_turn_vec = [0 for _ in range(max_tokens)]

        tmp_token_num = len(feat.tokens)
        tmp_turn_num = len(feat.speaker_vec)

        for x in range(tmp_token_num):
            tmp_utt_mask[x][:tmp_token_num] = feat.utt_mask[x]

        tmp_wv[:tmp_token_num] = feat.tokens
        tmp_pred[:tmp_token_num] = feat.pred_vec
        tmp_seg[:tmp_token_num] = feat.seg_vec
        tmp_speaker[:tmp_turn_num] = feat.speaker_vec
        tmp_turn_vec[:tmp_token_num] = feat.turn_vec
        tmp_label[:tmp_token_num] = feat.label
        tmp_mask[:tmp_token_num] = feat.input_mask
        tmp_cls[:tmp_turn_num] = feat.cls_position
        tmp_u_la[:tmp_turn_num] = feat.utt_labels
        tmp_last_label[:tmp_token_num] = feat.last_turn_label

        pad_tokens.append(tmp_wv)
        pad_pred.append(tmp_pred)
        pad_seg.append(tmp_seg)
        pad_speaker.append(tmp_speaker)
        pad_label.append(tmp_label)
        pad_mask.append(tmp_mask)
        pad_cls.append(tmp_cls)
        pad_utt_labels.append(tmp_u_la)
        pad_utt_mask.append(tmp_utt_mask)
        pad_last_label.append(tmp_last_label)
        pad_turn.append(tmp_turn_vec)

    return torch.tensor(pad_tokens).long(), torch.tensor(pad_pred).long(), torch.tensor(pad_seg).long(),\
           torch.tensor(pad_speaker).long(), torch.tensor(pad_turn).long(), torch.tensor(pad_mask).float(),\
           torch.tensor(pad_label).long(), torch.tensor(pad_cls).long(), torch.tensor(pad_utt_labels).long(),\
           torch.tensor(pad_utt_mask).float(), torch.tensor(pad_last_label).long()
