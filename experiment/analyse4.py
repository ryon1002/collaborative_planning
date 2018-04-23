import pickle
from world import graph_world2
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from analyse import read_result

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

human = read_result.read_result("result/result.txt")

bottom = np.zeros((11, 10, 4))
for beta in range(0, 11):
    for eval_id in range(1, 11):
        world = pickle.load(open("dump/sample_world_" + str(eval_id) + ".pkl", "r"))
        wb= graph_world2.GraphWorld(world, 0.4, 0.1 * beta)
        _, b, _ = wb.check_action(wb.base_action)
        bottom[beta, eval_id - 1] = b[answer_index[eval_id]]
    bottom[beta] = bottom[beta] / np.sum(bottom[beta] , axis=1, keepdims=True)

tasks = []
for h in human:
    values = []
    for beta in range(0, 11):
        values.append(scipy.stats.pearsonr(h.flatten(), bottom[beta].flatten())[0])
    tasks.append(np.argmax(np.array(values)) * 0.01)
from matplotlib.ticker import  MultipleLocator
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.hist(tasks, rwidth=0.8)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel(r"Best value of $\beta_3$", fontsize=16)
plt.ylabel("Number of participants", fontsize=16)
plt.savefig("result_3.eps", bbox_inches="tight", format="eps")
plt.show()

