import json
import numpy as np

user_data = [json.loads(l) for l in open("data.txt", "r")]

answer_type = {1: np.array([1, 2, 0, 0]),
               2: np.array([1, 3, 2, 3]),
               3: np.array([0, 1, 2, 0]),
               4: np.array([2, 3, 3, 1]),
               5: np.array([3, 1, 2, 3]),
               6: np.array([0, 2, 1, 0]),
               7: np.array([1, 2, 3, 0]),
               8: np.array([0, 1, 2, 3]),
               9: np.array([0, 2, 1, 3]),
               }

def analyze1(a_num, data):
    a_type = answer_type[a_num]
    data = np.array([int(d) for d in data])
    if np.max(data) - np.min(data) <= 1:
        return -1

    max_idx = data == np.max(data)
    if np.sum(max_idx) == 1:
        return a_type[max_idx][0]
    elif np.sum(max_idx) == 2:
        max_type = a_type[max_idx]
        if np.product(max_type) == 2 and np.sum(max_type) == 3:
            return 4
        if np.sum(max_type) == 0:
            return 0
        if np.prod(max_type) == 0:
            return 7
        if np.sum(max_type) == 4 or np.sum(max_type) == 5:
            return np.sum(max_type) + 1
    elif np.sum(max_idx) == 3:
        max_type = a_type[max_idx]
        if np.prod(max_type) == 0:
            return 7
        if np.sum(max_type) == 6:
            return 8
        if np.sum(max_type) == 7 or np.sum(max_type) == 8:
            return np.sum(max_type) + 2

format_data = []
for d in user_data:
    tmp_data = {}
    for a, result in sorted(d.viewitems()):
        tmp_data[int(a)] = analyze1(int(a), result)
    format_data.append(tmp_data)

for d in format_data:
    # print d
    # print [v for k, v in sorted(d.viewitems())]
    print ",".join([str(v) for k, v in sorted(d.viewitems())])
