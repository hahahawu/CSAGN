from dataset_utils import load_file
import tqdm
import argparse
import codecs
from utils import fine_print_res
from eval.eval_dialogue_srl import calc_f1
import copy
from predictor import Predictor

srl_model_dir = "exp"


def is_correct(gold_structure, pred_structure):
    for key in gold_structure:
        if key not in pred_structure:
            return False
        if gold_structure[key] != pred_structure[key]:
            return False

    return True


def evl(config):
    dataset_name = config.dataset
    if dataset_name == "duconv":
        datapath = "model_data/duconv.test.data"
    elif dataset_name == "news_dialog":
        datapath = "model_data/newsdialog.test.data"
    elif dataset_name == "personal_dialog":
        datapath = "model_data/personal.test.data"
    else:
        raise ValueError("You must choose the dataset from [duconv, news_dialog, personal_dialog]")

    model_name = config.model_name
    srl_model_path = "exp/{}".format(model_name)
    seg_type_id_map = {"[CLS]": 0, "[SEP]": 0, "agent": 2, "human": 3}

    srl_helper = Predictor(srl_model_path, config, seg_type_id_map)

    gold_all_srl_list = []
    pred_all_srl_list = []

    gold_inter_srl_list = []
    pred_inter_srl_list = []

    gold_inner_srl_list = []
    pred_inner_srl_list = []

    with codecs.open("result/{}-{}-result.all.comp.txt".format(model_name, dataset_name), 'w') as all_fw, \
            codecs.open("result/{}-{}-result.inter.comp.txt".format(model_name, dataset_name), 'w') as inter_fw, \
            codecs.open("result/{}-{}-result.inner.comp.txt".format(model_name, dataset_name), 'w') as inner_fw:
        data = load_file(datapath)

        if config.max_size > 0:
            data = data[:config.max_size]

        for i, (sent_id, pred_idx, words, dialog_turns, label) in tqdm.tqdm(enumerate(data)):
            input = copy.deepcopy(words)
            tags, reverse_mapping = srl_helper.predict(words, pred_idx, dialog_turns)

            gold_role_spans, gold_inner_role_spans, gold_inter_role_spans, _ = fine_print_res(input, label)
            pred_role_spans, pred_inner_role_spans, pred_inter_role_spans, _ = fine_print_res(input, tags,
                                                                                              reverse_mapping)

            gold_all_srl_list.append(gold_role_spans)
            pred_all_srl_list.append(pred_role_spans)
            gold_inter_srl_list.append(gold_inter_role_spans)
            pred_inter_srl_list.append(pred_inter_role_spans)
            gold_inner_srl_list.append(gold_inner_role_spans)
            pred_inner_srl_list.append(pred_inner_role_spans)

            write_comp_file(sent_id, gold_role_spans, pred_role_spans, input, pred_idx, all_fw)
            write_comp_file(sent_id, gold_inner_role_spans, pred_inner_role_spans, input, pred_idx, inner_fw)
            write_comp_file(sent_id, gold_inter_role_spans, pred_inter_role_spans, input, pred_idx, inter_fw)

        overall_f1 = calc_f1(gold_all_srl_list, pred_all_srl_list)
        inter_f1 = calc_f1(gold_inter_srl_list, pred_inter_srl_list)
        inner_f1 = calc_f1(gold_inner_srl_list, pred_inner_srl_list)
        print("results on all args: {}".format(overall_f1))
        print("results on inter args: {}".format(inter_f1))
        print("results on inner args: {}".format(inner_f1))

    return overall_f1['F']


def evl_detailed_res(gold_srl_list, pred_srl_list):
    labels = set()
    for srl in gold_srl_list:
        for label in srl:
            labels.add(label)

    for label in labels:
        gold_list = []
        pred_list = []

        for idx, gold_srl in enumerate(gold_srl_list):
            pred_srl = pred_srl_list[idx]

            new_srl = {}
            if label in gold_srl:
                new_srl[label] = gold_srl[label]
            new_srl["V"] = gold_srl["V"]

            gold_list.append(new_srl)

            new_srl = {}
            if label in pred_srl:
                new_srl[label] = pred_srl[label]
            new_srl["V"] = pred_srl["V"]

            pred_list.append(new_srl)

        print("results on {}: {}".format(label, calc_f1(gold_list, pred_list)))


def count_args_size(srl_list):
    res = 0
    for srl in srl_list:
        if "V" in srl:
            res += len(srl.keys()) - 1
        else:
            res += len(srl.keys())
    return res


def write_comp_file(sent_id, gold_spans, pred_spans, words, pred_idx, fw):
    for _ in range(len(words)):
        word = words[_]
        if word == "human" or word == "agent":
            if _ != 0:
                fw.write("\n")
        fw.write(word + "(" + str(_) + ")")
    fw.write("\n")

    fw.write("===============Sent Id===============" + "\n")
    fw.write(str(sent_id) + "\n\n")

    fw.write("===============Gold===============" + "\n")
    for role in gold_spans:
        if role == "V":
            fw.write(role + "-" + str(pred_idx))
        else:
            fw.write(role)
        fw.write(" :\t" + gold_spans[role] + "\n")
    fw.write("\n")

    fw.write("===============Pred===============" + "\n")
    for role in pred_spans:
        if role == "V":
            fw.write(role + "-" + str(pred_idx))
        else:
            fw.write(role)
        fw.write(" :\t" + pred_spans[role] + "\n")
    if is_correct(gold_spans, pred_spans):
        fw.write("Correct !!!\n")
    else:
        fw.write("Incorrect !!!\n")
    fw.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset", type=str, default="duconv")
    argparser.add_argument("model_name", type=str, help="please specify the name of the model")
    argparser.add_argument("-use_seg", action="store_true", default=False)
    argparser.add_argument("-max_size", type=int, default=-1)
    argparser.add_argument("-wp", type=int, default=10)
    argparser.add_argument("-wf", type=int, default=0)
    argparser.add_argument("-mode", choices=['sum', 'mean', 'max'], default='max')
    config = argparser.parse_args()
    evl(config)
