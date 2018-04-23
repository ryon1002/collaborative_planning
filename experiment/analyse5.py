import json
import itertools
import pickle
from world import graph_world2
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

# user_prof = {l.split("\t")[0]:l.split("\t")[1] for l in open("analize/result2.txt", "r")}

np.set_printoptions(edgeitems=3200, linewidth=10000, precision=3)
user_data = {l.split("\t")[0]:json.loads(l.split("\t")[1]) for l in open("result/result.txt", "r")}

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

answer_type = {1: np.array([1, 2, 0, 0]),
               2: np.array([1, 3, 2, 3]),
               3: np.array([0, 1, 2, 0]),
               4: np.array([2, 3, 3, 1]),
               5: np.array([3, 1, 2, 3]),
               6: np.array([0, 2, 1, 0]),
               7: np.array([1, 2, 3, 0]),
               8: np.array([0, 1, 2, 3]),
               9: np.array([0, 2, 1, 3]),
               10: np.array([1, 2, 0, 0]),
               }

format_data = []
for d in user_data.viewvalues():
    tmp_data = {}
    for a, (top_i, result) in sorted(d.viewitems()):
        tmp_data[int(a)] = [int(top_i), np.array([int(dd) for dd in result], dtype=np.float)]
    format_data.append(tmp_data)

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

# check_data = {}
# for k, v in user_data.viewitems():
#     tmp_data = {}
#     for a, (top_i, result) in sorted(v.viewitems()):
#         tmp_data[int(a)] = [int(top_i), np.array([int(dd) for dd in result], dtype=np.float)]
#     check_data[k] = (tmp_data)
# for k, v in check_data.viewitems():
#     if not is_wrong_user(v):
#         if k in user_prof:
#             print user_prof[k]
# exit()

format_data = [u for u in format_data if not is_wrong_user(u)]
# print len(format_data)
# exit()

def prob1(list, diff=0):
    return list[1] - diff / np.sum(list[1] - diff)

def is_wrong_answer(agent, answer):
    not_min = answer != np.min(answer)
    return np.prod(answer_type[agent][not_min]) == 0
    # answer_type[agent] =

def conv_to_prob(data):
    ret_prob = {k:prob1(v) for k, v in data.viewitems()}
    # ret_prob.pop(10)
    return ret_prob

def conv_to_prob2(data):
    d_list = [d for d in itertools.chain(*[v[1] for v in data.viewvalues()])]
    print d_list
    return {k:prob1(v[1], min(d_list)) for k, v in data.viewitems()}

def conv_to_prob3(data):
    d_list = [d for d in itertools.chain(*data.viewvalues())]
    return {k:prob1(v, min(d_list)) for k, v in data.viewitems() if not is_wrong_answer(k, v)}

prob_data = [{k:v[1] for k, v in u.viewitems()} for u in format_data]
human = np.zeros((len(prob_data), 10, 4))
for i, d in enumerate(prob_data):
    for k, v in d.viewitems():
        human[i][k - 1] = v


# top = np.zeros((11, 10, 4))
bottom = np.zeros((11, 10, 4))
# bottom2 = np.zeros((11, 10, 4))
for beta in range(0, 11):
    for eval_id in range(1, 11):
        world = pickle.load(open("dump/sample_world_" + str(eval_id) + ".pkl", "r"))
        wb= graph_world2.GraphWorld(world, 0.4, 0.1 * beta)
        _, b, _ = wb.check_action(wb.base_action)
        bottom[beta, eval_id - 1] = b[answer_index[eval_id]]
    bottom[beta] = bottom[beta] / np.sum(bottom[beta] , axis=1, keepdims=True)

print bottom
# for h in human:
#     values = []
#     for beta in range(0, 11):
#         values.append(scipy.stats.pearsonr(h.flatten(), bottom[beta].flatten())[0])
#     print np.argmax(np.array(values))


task_bb = []
task_b = []
task_t = []
for h in human:
    values = []
    for beta in range(0, 11):
        values.append(scipy.stats.pearsonr(h.flatten(), bottom[beta].flatten())[0])
    task_bb.append(np.max(np.array(values)))
    task_t.append(values[0])
    task_b.append(values[7])
print sum(task_bb) / len(human)
print sum(task_b) / len(human)
print sum(task_t) / len(human)
# exit()

for h in human:
    task = []
    for eval_id in range(1, 11):
        values = []
        for beta in range(0, 11):
            values.append(scipy.stats.pearsonr(h[eval_id - 1].flatten(), bottom[beta][eval_id - 1].flatten())[0])
        task.append(np.argmax(np.array(values)) * 0.01)
    print task
    # print np.std(task)
exit()

# print top
# print bottom
# print bottom2
# # exit()
# print human
exit()

# prob_data = [conv_to_prob(u) for u in format_data]
# prob_data = [conv_to_prob2(u) for u in format_data]
# format_data = [conv_to_prob3(u) for u in format_data]
# print prob_data
# exit()
# human = np.zeros((10, 4))
# for d in prob_data:
#     for k, v in d.viewitems():
#         human[k - 1] += v
# human = human / np.sum(human, axis=1, keepdims=True)
# print human
# print scipy.stats.pearsonr(human.flatten(), bottom.flatten())[0]
# exit()
# print len(prob_data)

# all user pearson
label = ["all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
label = ["2-1-2", "3-1-2", "3-2-2", "4-1-2", "4-2-2", "4-3-2", "2-1-3", "3-2-3", "4-3-3"]
##################### Total ######################
top_p = []
bottom_p = []
bottom2_p = []
# top_p.append(scipy.stats.pearsonr(human.flatten(), top.flatten())[0])
# bottom_p.append(scipy.stats.pearsonr(human.flatten(), bottom.flatten())[0])
print scipy.stats.pearsonr(human.flatten(), top.flatten())[0]
print scipy.stats.pearsonr(human.flatten(), bottom.flatten())[0]
print scipy.stats.pearsonr(human.flatten(), bottom2.flatten())[0]

for i in range(9):
    top_p.append(scipy.stats.pearsonr(human[i], top[i])[0])
    bottom_p.append(scipy.stats.pearsonr(human[i], bottom[i])[0])
    bottom2_p.append(scipy.stats.pearsonr(human[i], bottom2[i])[0])
##################### Total ######################


top_p = np.array(top_p)
bottom_p = np.array(bottom_p)
bottom2_p = np.array(bottom2_p)

# cor = 0.0
# i_index = (([0, 1, 3], [2, 3, 4]), ([2, 5], [3, 4]))
# i_index = (([0, 2, 5], [1, 2, 3]), ([1, 4], [1, 2]))
# i_index = (([0, 2, 5, 1, 4, 3], [1, 1, 1, 2, 2, 3]), )
# i_index = (([0, 2, 5, 1, 4, 3], [1, 1, 1, 2, 2, 3]), )
# for ind in i_index:
#     cor += scipy.stats.pearsonr(top_p[ind[0]], ind[1])[0]
# print cor / len(i_index)
# print cor
# i_index = [2, 3, 3, 4, 4, 4, 2, 3, 4] # k
# i_index = [2, 2, 2, 2, 2, 2, 3, 3, 3] # c
# i_index = [1, 1, 2, 1, 2, 3, 1, 2, 3] # p
i_index = [1, 2, 1, 3, 2, 1, 1, 1, 1] # p
print scipy.stats.pearsonr(top_p, i_index)[0]
print scipy.stats.pearsonr(bottom_p, i_index)[0]
print scipy.stats.pearsonr(bottom2_p, i_index)[0]

# i_index = [0, 1, 3], [2, 3, 4]
# i_index = [0, 6], [2, 3]
# print scipy.stats.pearsonr(top_p[i_index[0]], i_index[1])[0]
# i_index = [1, 7], [2, 3]
# print scipy.stats.pearsonr(top_p[i_index[0]], i_index[1])[0]
# i_index = [2, 8], [2, 3]
# # i_index = [0, 2, 5], [2, 3, 4]
# print scipy.stats.pearsonr(top_p[i_index[0]], i_index[1])[0]
# exit()

##################### Each ######################
# top_p = np.zeros((9), dtype=np.float)
# bottom_p = np.zeros((9), dtype=np.float)
# c_top_p = np.zeros((9), dtype=np.int)
# c_bottom_p = np.zeros((9), dtype=np.int)
# # top_p.append(scipy.stats.pearsonr(human.flatten(), top.flatten())[0])
# # bottom_p.append(scipy.stats.pearsonr(human.flatten(), bottom.flatten())[0])
# for data in prob_data:
#     for k, v in sorted(data.viewitems()):
#         if k == 10:
#             continue
#         t = scipy.stats.pearsonr(v, top[k-1])[0]
#         if not np.isnan(t):
#             top_p[k - 1] += t
#             c_top_p[k - 1] += 1
#         b = scipy.stats.pearsonr(v, bottom[k-1])[0]
#         if not np.isnan(b):
#             bottom_p[k - 1] += b
#             c_bottom_p[k - 1] += 1
# top_p /= c_top_p
# bottom_p /= c_bottom_p

##################### Each ######################
print top_p, bottom_p, label
index = np.arange(9)
# plt.bar(index-0.2, top_p, color="b", width=0.4, label="Full Bayes")
# plt.bar(index+0.2, bottom_p, color="g", width=0.4, label="Plan Prediction Oriented")
plt.bar(index-0.3, top_p, color="b", width=0.3, label="Full Bayes")
plt.bar(index, bottom_p, color="g", width=0.3, label="Plan Prediction Oriented")
plt.bar(index+0.3, bottom2_p, color="r", width=0.3, label="Plan Prediction Oriented")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(index, label)
plt.xlabel("k-n-c")
plt.ylabel("Pearson Correlation")
plt.legend(loc="lower right")
plt.savefig("result_1.eps", bbox_inches="tight", format="eps")
plt.show()
exit()

## per_user_pearson
top_flat = top.flatten()
bottom_flat=bottom.flatten()
t = []
b = []
for data in prob_data:
    dt = []
    db = []
    for k, v in sorted(data.viewitems()):
        dt.append(scipy.stats.pearsonr(v, top[k-1])[0])
        db.append(scipy.stats.pearsonr(v, bottom[k-1])[0])
    dt = np.array(dt)
    dt[np.isnan(dt)] = 0
    db = np.array(db)
    db[np.isnan(db)] = 0
    t.append(np.mean(np.array(dt)))
    b.append(np.mean(np.array(db)))
    # d_list = [d for d in itertools.chain(*[v for k,v in sorted(data.viewitems())])]
    # t.append(scipy.stats.pearsonr(d_list, top_flat)[0])
    # b.append(scipy.stats.pearsonr(d_list, bottom_flat)[0])
print t
print b
print len(t)
baseline = np.arange(-0.2, 1.2, 0.1)
plt.scatter(t, b)
plt.plot(baseline, baseline, "--", linewidth=0.5, color="k")
plt.ylabel("Plan Prediction Oriented")
plt.xlabel("Full Bayes")
plt.savefig("result_2.eps", bbox_inches="tight", format="eps")
plt.show()
exit()

## per_user_pearson
# top_flat = top.flatten()
# bottom_flat=bottom.flatten()
# human_flat=human.flatten()
# plt.scatter(human_flat, top_flat, label="top")
# plt.scatter(human_flat, bottom_flat, label="bottom")
# plt.legend(loc="best")
# plt.show()
# exit()

top_idx = {k:np.where(v==1)[0][0] for k,v in answer_type.viewitems()}
bottom_idx = {k:np.where(v==2)[0][0] for k,v in answer_type.viewitems()}
t_count ={k:0 for k in range(1, 11)}
b_count ={k:0 for k in range(1, 11)}
for data in format_data:
    for k, v in sorted(data.viewitems()):
        if top_idx[k] + 1 == v[0]:
            t_count[k] += 1
        if bottom_idx[k] + 1 == v[0]:
            b_count[k] += 1
index = np.arange(10)
plt.bar(index-0.2, [float(v) / len(format_data) for k, v in sorted(t_count.viewitems())], color="b", width=0.4, label="top")
plt.bar(index+0.2, [float(v) / len(format_data) for k, v in sorted(b_count.viewitems())], color="g", width=0.4, label="bottom")
label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
plt.xticks(index, label)
plt.legend(loc="best")
plt.show()
exit()

top_idx = {k:np.where(v==1)[0][0] for k,v in answer_type.viewitems()}
bottom_idx = {k:np.where(v==2)[0][0] for k,v in answer_type.viewitems()}
t_count ={k:0 for k in range(1, 11)}
b_count ={k:0 for k in range(1, 11)}
top_list = []
bottom_list = []
for data in format_data:
    top, bottom = 0.0, 0.0
    for k, v in sorted(data.viewitems()):
        if k == 10:
            continue
        if top_idx[k] + 1 == v[0]:
            top += 1
        if bottom_idx[k] + 1 == v[0]:
            bottom += 1
    top_list.append(top / 9)
    bottom_list.append(bottom / 9)
print top_list
print bottom_list

# index = np.arange(10)
# plt.bar(index-0.2, [float(v) / len(format_data) for k, v in sorted(t_count.viewitems())], color="b", width=0.4, label="top")
# plt.bar(index+0.2, [float(v) / len(format_data) for k, v in sorted(b_count.viewitems())], color="g", width=0.4, label="bottom")
# label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
# plt.xticks(index, label)
# plt.scatter(top_list, bottom_list)
# plt.legend(loc="best")
# plt.show()
# plt.hist(bottom_list)
# plt.show()
# exit()
