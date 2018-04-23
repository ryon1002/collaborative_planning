import itertools
import json
import numpy as np


def is_wrong_user(data):
    d_list = [d for d in itertools.chain(*[v[1] for v in data.viewvalues()])]
    if max(d_list) - min(d_list) <= 1:
        return True
    wrong_answer = 0
    for k, v in data.viewitems():
        if k == 10:
            continue
        if np.argmin(v[1]) + 1 == v[0]:
            wrong_answer += 1
            if wrong_answer == 2:
                return True
    return False


def read_result(filename, type=None):
    format_data = []
    user_data = {l.split("\t")[0]: json.loads(l.split("\t")[1]) for l in open(filename, "r")}
    for d in user_data.viewvalues():
        tmp_data = {}
        for a, (top_i, result) in sorted(d.viewitems()):
            tmp_data[int(a)] = [int(top_i), np.array([int(dd) for dd in result], dtype=np.float)]
        format_data.append(tmp_data)

    format_data = [u for u in format_data if not is_wrong_user(u)]
    prob_data = [{k: v[1] for k, v in u.viewitems()} for u in format_data]
    human = np.zeros((len(prob_data), 10, 4))
    for i, d in enumerate(prob_data):
        for k, v in d.viewitems():
            human[i][k - 1] = v
    if type == "mean":
        return np.mean(human, axis=0)
    return human
