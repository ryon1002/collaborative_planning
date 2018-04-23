import pickle
from world import graph_world2
import numpy as np


def calc_model_result():
    answer_index = {1: [0, 1, 2, 3],
                    2: [0, 1, 2, 3],
                    3: [1, 2, 3, 6],
                    4: [8, 9, 13, 15],
                    5: [8, 9, 10, 11],
                    6: [5, 10, 11, 12],
                    7: [0, 1, 2, 3],
                    8: [2, 3, 4, 5],
                    9: [8, 15, 16, 17],
                    10: [0, 1, 2, 3],
                    }

    top = np.zeros((10, 4))
    bottom = np.zeros((10, 4))

    for eval_id in range(1, 11):
        world = pickle.load(open("../experiment/dump/sample_world_" + str(eval_id) + ".pkl", "r"))
        wt = graph_world2.GraphWorld(world, 0.5)
        wb = graph_world2.GraphWorld(world, 0.5, 0.5)
        t, _, _ = wt.check_action(wt.base_action)
        _, b, _ = wb.check_action(wb.base_action)
        top[eval_id - 1] = t[answer_index[eval_id]]
        bottom[eval_id - 1] = b[answer_index[eval_id]]

    top = top / np.sum(top, axis=1, keepdims=True)
    bottom = bottom / np.sum(bottom, axis=1, keepdims=True)
    return top, bottom
