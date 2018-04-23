import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def calc_pearson_correlation(human, top, bottom):
    # all user pearson
    print scipy.stats.pearsonr(human.flatten(), top.flatten())[0]
    print scipy.stats.pearsonr(human.flatten(), bottom.flatten())[0]
    label = ["2-1-2", "3-1-2", "3-2-2", "4-1-2", "4-2-2", "4-3-2", "2-1-3", "3-2-3", "4-3-3"]
    ##################### Total ######################
    top_p = []
    bottom_p = []

    for i in range(9):
        top_p.append(scipy.stats.pearsonr(human[i], top[i])[0])
        bottom_p.append(scipy.stats.pearsonr(human[i], bottom[i])[0])
    ##################### Total ######################

    top_p = np.array(top_p)
    bottom_p = np.array(bottom_p)

    index = np.arange(9)
    plt.bar(index - 0.2, top_p, color="b", width=0.4, label="Full Inverse Planning")
    plt.bar(index + 0.2, bottom_p, color="g", width=0.4, label="Plan Predictability Oriented")
    plt.xticks(index, label)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("(k,n,c)", fontsize=16)
    plt.ylabel("Pearson Correlation", fontsize=16)
    plt.legend(loc="lower right", bbox_to_anchor=(1.015, 1), fontsize=12)
    plt.savefig("result_1.eps", bbox_inches="tight", format="eps")
    plt.show()
