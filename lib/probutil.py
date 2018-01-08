import numpy as np


def normalized_array(array):
    return array / np.sum(array)


def normalized_2d_array(array, axis):
    return array / np.sum(array, axis=axis, keepdims=True)


def make_prob_dist(objective, beta=1, top_filter=0):
    prob = np.exp(beta * objective)
    if top_filter > 0:
        low_index = prob.argsort()[::-1][top_filter:]
        prob[low_index] = 0
    return normalized_array(prob)


def make_prob_2d_dist(objective, axis, beta=1):
    return normalized_2d_array(np.exp(beta * objective), axis=axis)
