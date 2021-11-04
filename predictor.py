import torch
from dataset_utils import bert_vectorize_data
import numpy as np
from transformers import BertTokenizer, BertConfig
from Config import configure
import os
from dataset_utils import load_vocab
from model import CSAGN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Predictor(object):
    def __init__(self, model_path, config, seg_type_id_map):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.device = device
        print("===================== Configure =====================")
        for key in configure:
            print("{}: {}".format(key, configure[key]))

        self.seg_type_id_map = seg_type_id_map

        label_vocab_path = configure["label_vocab_path"]
        label_vocab = {"O": 0}
        label_vocab["[SEP]"] = len(label_vocab)
        label_vocab["[CLS]"] = len(label_vocab)
        label_vocab = load_vocab(label_vocab_path, label_vocab)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)

        model_config = BertConfig.from_pretrained(model_path, type_vocab_size=len(seg_type_id_map),
                                                  num_labels=len(label_vocab))

        model = CSAGN.from_pretrained(model_path, config=model_config, wp=config.wp, wf=config.wf).to(device)
        model.eval()

        self.model = model

        self.label_vocab = label_vocab
        self.label_vocab_size = len(self.label_vocab)
        self.idx_to_label = dict(zip(label_vocab.values(), label_vocab.keys()))
        self.use_seg = config.use_seg

    def predict(self, words, pred_vec, dialog_turns):
        sen_vec, seg_vec, pred_vec, input_mask, bert_word_len, speaker_vec, turn_vec, cls_pos, utt_mask = bert_vectorize_data(
            self.bert_tokenizer, (pred_vec, words, dialog_turns), self.label_vocab, self.seg_type_id_map, "test", self.use_seg)
        while len(sen_vec) >= 512:
            top_word_len = bert_word_len.pop(1)
            for _ in range(top_word_len):
                drop_w = sen_vec.pop(1)
                seg_vec.pop(1)
                turn_vec.pop(1)
                pred_vec.pop(1)
                input_mask.pop(1)
                utt_mask.pop(1)
                for m in utt_mask:
                    m.pop(1)
                if drop_w in self.bert_tokenizer.convert_tokens_to_ids(["agent", "human"]):
                    speaker_vec.pop(0)
                    cls_pos.pop(0)
                    turn_vec.pop(0)
                cls_pos = [cp-1 for cp in cls_pos]

        device = self.device
        seq_len = len(sen_vec)

        lengths = torch.tensor([len(speaker_vec)]).long().to(device)

        sen_vec = torch.from_numpy(np.array(sen_vec)).long().to(device).view(1, -1)
        seg_vec = torch.from_numpy(np.array(seg_vec)).long().to(device).view(1, -1)
        pred_vec = torch.from_numpy(np.array(pred_vec)).long().to(device).view(1, -1)
        input_mask = torch.from_numpy(np.array(input_mask)).float().to(device).view(1, -1)
        cls_pos = torch.from_numpy(np.array(cls_pos)).long().to(device).view(1, -1)
        speaker_vec = torch.tensor(np.array(speaker_vec)).long().to(device).view(1, -1)
        turn_vec = torch.tensor(np.array(turn_vec)).long().to(device).view(1, -1)
        utt_mask = torch.tensor(np.array(utt_mask)).float().to(device).view(1, seq_len, seq_len)

        loss, predicted = self.model(input_ids=sen_vec, token_type_ids=seg_vec, attention_mask=input_mask,
                                     pred_ids=pred_vec, text_lens=lengths, speaker_ids=speaker_vec, cls_vec=cls_pos,
                                     utt_mask=utt_mask, turn_ids=turn_vec)
        pred_seq = predicted.contiguous().view(-1).cpu().numpy()

        res = [self.idx_to_label[idx] for idx in pred_seq]
        res = res[1:-1]
        bert_word_len = bert_word_len[1:-1]

        reverse_mapping = []
        for input_idx, l in enumerate(bert_word_len):
            for j in range(l):
                reverse_mapping.append(input_idx)

        return res, reverse_mapping
