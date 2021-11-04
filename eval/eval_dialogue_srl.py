
import os, sys, json
from collections import Counter

def load_data(path):
    data = {}
    for line in open(path, 'r'):
        jobj = json.loads(line.strip())
        sentid = jobj['sentid']
        assert sentid not in data
        data[sentid] = []
        conversation = jobj['sent'].replace('<SEP>', '', 100).split()
        for pa_structure in jobj['srl']:
            pas = {'V': conversation[pa_structure['pred']]}
            for k, v in pa_structure['args'].items():
                st, ed = v
                if ed == -1:
                    pas[k] = '我'
                elif ed == -2:
                    pas[k] = '你'
                else:
                    pas[k] = ' '.join(conversation[st:ed+1])
            data[sentid].append(pas)
    return data


def update_counts_intersect(v1, v2, is_token_level):
    if v1 == '' or v2 == '':
        return 0
    if is_token_level:
        v1 = Counter(v1.split())
        v2 = Counter(v2.split())
        res = 0
        for k, cnt1 in v1.items():
            if k in v2:
                res += min(cnt1, v2[k])
        return res
    else:
        return v1 == v2


def update_counts_denominator(conv, is_token_level):
    counts = 0
    for pas in conv:
        for k, v in pas.items():
            if k != 'V': # don't count "pred" for each PA structure
                counts += len(v.split()) if is_token_level else 1
    return counts


# is_sync: whether ref-file and prd-file have the same content. This is always Ture except when the prd-file is after rewriting.
def update_counts(ref_conv, prd_conv, counts, is_sync, is_token_level):
    counts[1] += update_counts_denominator(ref_conv, is_token_level)
    counts[2] += update_counts_denominator(prd_conv, is_token_level)
    if is_sync:
        for ref_pas, prd_pas in zip(ref_conv, prd_conv):
            for k, v1 in ref_pas.items():
                if k == 'V':
                    continue
                v2 = prd_pas.get(k,'')
                counts[0] += update_counts_intersect(v1, v2, is_token_level)
    else:
        for ref_pas in ref_conv:
            for prd_pas in prd_conv:
                if prd_pas['V'] == ref_pas['V']:
                    for k, v1 in ref_pas.items():
                        if k == 'V':
                            continue
                        v2 = prd_pas.get(k,'')
                        counts[0] += update_counts_intersect(v1, v2, is_token_level)
                    break


def calc_f1(ref, prd, is_sync=True, is_token_level=False):
    """
    :param ref: a list of predicate argument structures
    :param prd:
    :return:
    """
    counts = [0, 0, 0]
    update_counts(ref, prd, counts, is_sync, is_token_level)
    p = 0.0 if counts[2] == 0 else counts[0]/counts[2]
    r = 0.0 if counts[1] == 0 else counts[0]/counts[1]
    f = 0.0 if p == 0.0 or r == 0.0 else 2*p*r/(p+r)
    return {'P':p, 'R':r, 'F':f}


if __name__ == "__main__":
    ref = load_data("../data/dev.txt")
    prd = load_data("../data/dev.txt")
    is_sync = True
    is_token_level = False

    ref_list = []
    prd_list = []
    for key, ref_data in ref.items():
        prd_data = prd.get(key, [])
        ref_list.extend(ref_data)
        prd_list.extend(prd_data)
    print(calc_f1(ref_list, prd_list, is_sync, is_token_level))
