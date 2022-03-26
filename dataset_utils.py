import copy


def load_file(path):
    with open(path, 'r', encoding='utf-8') as fr:
        res = []
        for line in fr:
            info = line.split("|||")
            sent_id = info[0]
            pred_idx = int(info[1].strip())
            words = info[2].strip().split(" ")
            dialog_turns = [int(x) for x in info[3].split(" ")]
            tags = info[4].split(" ")
            res.append((sent_id, pred_idx, words, dialog_turns, tags))
        return res


def load_vocab(path, symbol_idx=None):
    if symbol_idx is None:
        symbol_idx = {}
    with open(path, 'r', encoding="utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            if symbol not in symbol_idx:
                symbol_idx[symbol] = len(symbol_idx)
    return symbol_idx


def bert_vectorize_data(tokenizer, data, label_vocab, seg_type_id_map, mode="training", use_seg=False):
    data = copy.deepcopy(data)
    if mode == "training":
        sent_id, pred_idx, words, dialog_turns, labels = data
        label_vec = []
        labels.append("[SEP]")
        labels.insert(0, "[CLS]")
    else:
        pred_idx, words, dialog_turns = data

    segment_ids = []
    words.append("[SEP]")
    dialog_turns.append(100)

    pred_idx += 1
    words.insert(0, "[CLS]")
    dialog_turns.insert(0, 100)

    bert_word_len = []
    sen_vec = []
    pred_vec = []
    speaker_vec = []
    turn_vec = []
    cls_pos = []

    seg_type = "agent"

    for _ in range(len(words)):
        if words[_] == "agent":
            seg_type = "agent"
            speaker_vec.append(1)
            cls_pos.append(len(sen_vec))
        elif words[_] == "human":
            seg_type = "human"
            speaker_vec.append(2)
            cls_pos.append(len(sen_vec))

        if words[_] == "[CLS]" or words[_] == "[SEP]":
            tokens = [words[_]]
            seg_type = words[_]
        else:
            tokens = tokenizer.tokenize(words[_])

        bert_word_len.append(len(tokens))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        sen_vec.extend(ids)
        if use_seg:
            segment_ids.extend([seg_type_id_map[seg_type]] * len(ids))
        else:
            segment_ids.extend([0] * len(ids))

        assert len(sen_vec) == len(segment_ids)

        if mode == "training":
            if labels[_] in label_vocab:
                label_vec.append(label_vocab[labels[_]])
                if labels[_] == "O":
                    label_vec.extend([label_vocab["O"]] * (len(ids) - 1))  # assign label "X" for sub-word
                elif len(ids) > 1:
                    label_vec.extend((len(ids) - 1) * [label_vocab["I" + labels[_][1:]]])
            else:
                label_vec.extend([label_vocab["O"]] * len(ids))

            assert len(sen_vec) == len(label_vec)

        if _ == pred_idx:
            pred_vec.extend([2] * len(ids))
        else:
            pred_vec.extend([1] * len(ids))

        turn_num = dialog_turns[_]
        if turn_num == 100:
            turn_vec.extend([0] * len(ids))
        else:
            tmp_turn = min(dialog_turns[_] + 1, 10)
            turn_vec.extend([tmp_turn] * len(ids))

    # distance of duconv training set: {1: 14582, 2: 6103, 3: 1488, 4: 852, 5: 204, 6: 204, 7: 45, 8: 48, 9: 8, 10: 6}
    # cur:0 -> 1, neighboring turns: 1,2 -> 2; close turns: 3,4 -> 3;
    # middle-range turns: 5,6 -> 4; long-range turns: >=7 -> 5
    # turn_num = turn_vec[-1]
    # for _ in range(len(turn_vec)):
    #     if turn_num - turn_vec[_] >= 7:
    #         turn_vec[_] = 5
    #     elif turn_num - turn_vec[_] >= 5:
    #         turn_vec[_] = 4
    #     elif turn_num - turn_vec[_] >= 3:
    #         turn_vec[_] = 3
    #     elif turn_num - turn_vec[_] >= 1:
    #         turn_vec[_] = 2
    #     else:
    #         turn_vec[_] = 1

    tokens = tokenizer.convert_ids_to_tokens(sen_vec)
    assert len(sen_vec) == len(turn_vec)
    input_mask = [1 for _ in range(len(tokens))]

    utt_mask = [[0 for x in range(len(tokens))] for _ in range(len(tokens))]
    last_utt_idx = cls_pos[-1]

    for i, cls in enumerate(cls_pos[:-1]):
        e = cls_pos[i+1]
        for x in range(cls, e):
            utt_mask[x][cls:e] = [1] * (e - cls)
            utt_mask[x][last_utt_idx:-1] = [1] * (len(tokens) - 1 - last_utt_idx)

    for x in range(last_utt_idx, len(tokens)-1):
        utt_mask[x][last_utt_idx:-1] = [1] * (len(tokens) - 1 - last_utt_idx)

    utt_mask[0] = [1] * len(tokens)
    utt_mask[-1] = [1] * len(tokens)

    if mode == "training":
        utt_labels = []
        for x in range(len(cls_pos)):
            if x == len(cls_pos) - 1:
                s, e = cls_pos[x], len(label_vec) - 1
                tmp_labels = label_vec[s:e]
            else:
                s, e = cls_pos[x], cls_pos[x+1]
                tmp_labels = label_vec[s:e]
            if tmp_labels == [label_vocab["O"]] * (e - s):
                utt_labels.append(1)
            elif label_vocab["B-V"] in tmp_labels:
                utt_labels.append(3)
            else:
                utt_labels.append(2)
        last_turn_label = [-100 for _ in range(len(label_vec))]
        last_turn_label[last_utt_idx:-1] = label_vec[last_utt_idx:-1]
        assert len(last_turn_label) == len(label_vec)
        assert len(speaker_vec) == len(utt_labels)
        return sen_vec, segment_ids, pred_vec, input_mask, label_vec, speaker_vec, turn_vec, cls_pos, utt_labels, utt_mask, last_turn_label
    else:
        return sen_vec, segment_ids, pred_vec, input_mask, bert_word_len, speaker_vec, turn_vec, cls_pos, utt_mask
