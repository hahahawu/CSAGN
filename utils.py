import numpy as np
import math
import enum


def str2bool(v):
    if v.lower() in ('y', 'yes', 't', 'true', '1'):
        return True
    elif v.lower() in ('n', 'no', 'f', 'false', '0'):
        return False


def fine_print_res(words, labels, reverse_mapping=None):
    role_idx = {}
    for _ in range(len(labels)):
        label = labels[_]
        if label[0] == "B" or label[0] == "I" or label[0] == "V":
            if label[0] == "V":
                role = "V"
            else:
                role = label[2:]
            if role in role_idx:
                role_idx[role].append(_)
            else:
                role_idx[role] = [_]

    # make sure that the role idx list is in the order
    for role in role_idx:
        l = role_idx[role]
        if len(l) > 0:
            fi = l[0]
            li = l[-1]
            if li - fi + 1 != len(l):
                # we need to make sure to select the first occurred span
                if role != "V":
                    new_fi, new_li = is_include_multi_mentions(labels, fi, li)
                    fi = new_fi
                    li = new_li

                l = [t for t in range(fi, li + 1)]
                role_idx[role] = l

    if reverse_mapping is not None:
        for role in role_idx:
            input_left = reverse_mapping[role_idx[role][0]]
            input_right = reverse_mapping[role_idx[role][-1]] + 1
            role_idx[role] = [_ for _ in range(input_left, input_right)]

    sep_symbol = ""
    all_role_spans = {}
    inner_role_spans = {}
    inter_role_spans = {}
    for key in role_idx:
        all_role_spans[key] = sep_symbol.join([words[id] for id in role_idx[key]])

    if "V" not in role_idx or len(role_idx["V"]) == 0:
        inner_role_spans = all_role_spans
        inter_role_spans = all_role_spans
    else:
        pred_idx = role_idx["V"][0]
        for key in role_idx:
            val = sep_symbol.join([words[id] for id in role_idx[key]])
            if key == "V":
                inner_role_spans[key] = val
                inter_role_spans[key] = val
                continue
            tmp = role_idx[key]
            left_idx = tmp[0]
            right_idx = tmp[-1]
            if left_idx > pred_idx:
                inner_role_spans[key] = val
            elif right_idx < pred_idx:
                found_turn_break = False
                for _ in range(right_idx + 1, pred_idx):
                    if words[_] == "human" or words[_] == "agent":
                        found_turn_break = True
                        break
                if found_turn_break:
                    inter_role_spans[key] = val
                else:
                    inner_role_spans[key] = val

    return all_role_spans, inner_role_spans, inter_role_spans, role_idx


def is_include_multi_mentions(labels, from_idx, to_idx):
    is_find_mentions = False
    new_from_idx = -1
    new_to_idx = -1
    for _ in range(from_idx, to_idx + 1):
        if labels[_][0] == "B":
            if not is_find_mentions:
                is_find_mentions = True
                new_from_idx = _
                new_to_idx = _
            else:
                return new_from_idx, new_to_idx
        elif labels[_][0] == "I":
            new_to_idx = _
    return from_idx, to_idx


import torch


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_bin_path = path_prefix + "/pytorch_model.bin"
    model_config_path = path_prefix + "/config.json"

    torch.save(model_to_save.state_dict(), model_bin_path)
    model_to_save.bert.config.to_json_file(model_config_path)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    return torch.from_numpy(signal).long()


# Support for 3 different GAT implementations - we'll profile each one of these in playground.py
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2
