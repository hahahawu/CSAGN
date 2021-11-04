from dataset import SRLSet, collate_fn
from Config import configure
import numpy as np
import torch
from torch.utils import data
import os
from dataset_utils import load_vocab
import math
import argparse
import torch.nn as nn
from utils import save_model
from transformers import BertTokenizer, BertConfig, AdamW
import random
from utils import str2bool
from model import CSAGN
import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"
__CUDA__ = False
if torch.cuda.is_available():
    device = "cuda"
    __CUDA__ = True


def train():
    model.train()
    aveloss = 0
    count = 0
    for i, (sen_vec, pred_vec, seg_vec, speaker_vec, turn_vec, input_mask, label_vec, cls_vec, utt_labels, utt_mask,
            last_label) in enumerate(train_loader):
        sen_vec = sen_vec.to(device)  # (bsz, tokens)
        seg_vec = seg_vec.to(device)  # (bsz, tokens)
        pred_vec = pred_vec.to(device)  # (bsz, tokens)
        speaker_vec = speaker_vec.to(device)  # (bsz, turns)
        turn_vec = turn_vec.to(device)  # (bsz, turns)
        input_mask = input_mask.to(device)  # (bsz, tokens)
        label_vec = label_vec.to(device)  # (bsz, tokens)
        cls_vec = cls_vec.to(device)  # (bsz, turns)
        utt_labels = utt_labels.to(device)
        utt_mask = utt_mask.to(device)
        last_label = last_label.to(device)

        bsz = sen_vec.shape[0]
        lengths = torch.tensor([len(speaker_vec[_][speaker_vec[_] > 0].tolist()) for _ in range(bsz)])

        loss, predicted = model(input_ids=sen_vec, token_type_ids=seg_vec, attention_mask=input_mask, text_lens=lengths,
                                speaker_ids=speaker_vec, pred_ids=pred_vec, labels=label_vec, cls_vec=cls_vec,
                                utt_labels=utt_labels, utt_mask=utt_mask, last_label=last_label, turn_ids=turn_vec)

        loss = loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        aveloss += float(loss)
        count += 1

    return aveloss / count


def val():
    model.eval()
    aveloss = 0
    count = 0
    correct_count = 0
    total_count = 0
    total_non_zero_count = 0

    with torch.no_grad():
        for i, (sen_vec, pred_vec, seg_vec, speaker_vec, turn_vec, input_mask, label_vec, cls_vec, utt_labels, utt_mask,
                last_label) in enumerate(dev_loader):
            sen_vec = sen_vec.to(device)  # (bsz, tokens)
            seg_vec = seg_vec.to(device)  # (bsz, tokens)
            pred_vec = pred_vec.to(device)  # (bsz, tokens)
            speaker_vec = speaker_vec.to(device)  # (bsz, turns)
            turn_vec = turn_vec.to(device)  # (bsz, turns)
            input_mask = input_mask.to(device)  # (bsz, tokens)
            label_vec = label_vec.to(device)  # (bsz, tokens)
            cls_vec = cls_vec.to(device)  # (bsz, turns)
            utt_labels = utt_labels.to(device)
            utt_mask = utt_mask.to(device)
            last_label = last_label.to(device)

            bsz = sen_vec.shape[0]
            lengths = torch.tensor([len(speaker_vec[_][speaker_vec[_] > 0].tolist()) for _ in range(bsz)])

            loss, predicted = model(input_ids=sen_vec, token_type_ids=seg_vec, attention_mask=input_mask,
                                    text_lens=lengths, speaker_ids=speaker_vec, pred_ids=pred_vec, labels=label_vec,
                                    cls_vec=cls_vec, utt_labels=utt_labels, utt_mask=utt_mask, last_label=last_label,
                                    turn_ids=turn_vec)
            loss = loss.mean()

            label_vec = label_vec.contiguous().view(-1)
            predicted = predicted.contiguous().view(-1)

            non_zero_mask = torch.gt(predicted, valid_idx).float().to(device) * input_mask.contiguous().view(-1)
            eq_num = torch.eq(predicted, label_vec).float().to(device)
            eq_num = eq_num * non_zero_mask.contiguous().view(-1)
            eq_num = torch.sum(eq_num)

            aveloss += float(loss)
            count += 1

            correct_count += eq_num.cpu()
            total_count += torch.sum(non_zero_mask).cpu()
            total_non_zero_count += torch.sum(torch.gt(label_vec, valid_idx)).cpu()

    p = correct_count / total_count
    r = correct_count / total_non_zero_count

    return aveloss / count, p, r, 2 * p * r / (p + r)


if __name__ == "__main__":
    seg_type_id_map = {"[CLS]": 0, "[SEP]": 0, "agent": 2, "human": 3}

    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_name", type=str, help="please specify the name of the model")
    argparser.add_argument("-batch_size", type=int, default=configure["batch_size"])
    argparser.add_argument("-bert_version", type=str, default=configure["bert_version"])
    argparser.add_argument("-use_pretrain", type=str2bool, default=False)
    argparser.add_argument("-max_example_num", type=int, default=-1)
    argparser.add_argument("-use_seg", action="store_true", default=False, help="use segment embeddings.")
    argparser.add_argument("-wp", type=int, default=10, help="previous utterances window")
    argparser.add_argument("-wf", type=int, default=0, help="future utterances window")
    argparser.add_argument("-mode", choices=['sum', 'mean', 'max'], default='max', help="utterance pooling")
    argparser.add_argument("-intra_loss", action="store_true", default=False, help="True for adding intra-arg. loss")
    argparser.add_argument("-inter_loss", action="store_true", default=False, help="True for adding UTO loss")

    config = argparser.parse_args()

    configure["batch_size"] = config.batch_size
    configure["bert_version"] = config.bert_version
    wp = config.wp
    wf = config.wf
    mode = config.mode
    use_intra_loss = config.intra_loss
    use_inter_loss = config.inter_loss

    model_name = config.model_name
    use_pretrain = config.use_pretrain
    max_example_num = config.max_example_num
    use_seg = config.use_seg

    train_data_path = configure["train_data_path"]
    dev_data_path = configure["dev_data_path"]
    label_vocab_path = configure["label_vocab_path"]
    batch_size = configure["batch_size"]
    num_workers = configure["num_workers"]
    hidden_size = configure["hidden_size"]
    dropout = configure["dropout"]
    num_epochs = configure["num_epochs"]
    lr = configure["lr"]
    bert_version = configure["bert_version"]
    seed = configure["seed"]

    print("===================== Configure =====================")
    for key in configure:
        print("{}: {}".format(key, configure[key]))

    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    label_vocab = {"O": 0}
    label_vocab["[SEP]"] = len(label_vocab)
    label_vocab["[CLS]"] = len(label_vocab)
    valid_idx = 2
    label_vocab = load_vocab(label_vocab_path, label_vocab)
    label_vocab_size = len(label_vocab)

    worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_version)
    bert_tokenizer.add_tokens(["agent", "human"])

    train_set = SRLSet(train_data_path, bert_tokenizer, label_vocab, "training", seg_type_id_map,
                       max_size=max_example_num, use_seg=use_seg)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    dev_set = SRLSet(dev_data_path, bert_tokenizer, label_vocab, "training", seg_type_id_map, max_size=max_example_num,
                     use_seg=use_seg)
    dev_loader = data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    if not use_pretrain:
        model_config = BertConfig.from_pretrained(bert_version, num_labels=len(label_vocab))
        model = CSAGN.from_pretrained(
            bert_version, config=model_config, wp=wp, wf=wf, intra_loss=use_intra_loss,
            inter_loss=use_inter_loss).to(device)
        model.bert.resize_seg_type_embeddings(len(seg_type_id_map))
        model.bert.resize_token_embeddings(len(bert_tokenizer))
        model.bert.clone_embeddings()
    else:
        model_config = BertConfig.from_pretrained(bert_version, type_vocab_size=len(seg_type_id_map),
                                                  num_labels=len(label_vocab))
        model = CSAGN.from_pretrained(
            bert_version, config=model_config, wp=wp, wf=wf, intra_loss=use_intra_loss,
            inter_loss=use_inter_loss).to(device)
        model.bert.clone_embeddings()

    if n_gpu > 1:
        model = nn.DataParallel(model)

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # train_steps = num_epochs * int(math.ceil(len(train_set.instance) / batch_size))
    optimizer = AdamW(grouped_params, lr=lr)

    saved_model_dir = configure["model_base_dir"] + os.path.sep + model_name
    # tmp_model_dir = configure["model_base_dir"] + os.path.sep + "temp"

    best_dev_f1 = 0
    early_stop = 5
    no_increase = 0

    for iter in tqdm.tqdm(range(num_epochs)):
        train_loss = train()
        dev_loss, dev_acc, dev_rec, dev_token_f1 = val()

        if dev_token_f1 > best_dev_f1:
            no_increase = 0
            best_dev_f1 = dev_token_f1
            # save model
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            save_model(model, saved_model_dir)
            bert_tokenizer.save_pretrained(saved_model_dir)

            print("saved model to {}".format(saved_model_dir))
        else:
            no_increase += 1
            print("** * No improvements in last {} epoch. * **".format(no_increase))

        if no_increase == early_stop:
            print("** * No improvements in last 5 epochs. * **")
            break

        print("iter: {}, train loss: {}, dev loss: {}, dev_acc: {}, dev_recall: {}, dev_f: {}, best_f: {}".format(
            iter, train_loss, dev_loss, dev_acc, dev_rec, dev_token_f1, best_dev_f1))
