import numpy as np
import matplotlib.pyplot as plt
# from analyse import read_result
# import calc_model_result
#
# human = read_result.read_result("result/result.txt")
# top, bottom = calc_model_result.calc_model_result()

def compare_result(human, top, bottom, result_index):
    index = np.arange(4)
    fig, ax1 = plt.subplots()
    label=["(a)", "(b)", "(c)", "(d)"]
    ax1.bar(index-0.3, top[result_index], color="b", width=0.3, label="Full Inverse Planning")
    ax1.bar(index, bottom[result_index], color="g", width=0.3, label="Plan Predictability Oriented")
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Candidate", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend(loc="lower left", bbox_to_anchor=(-0.015, 1), fontsize=11.5)
    ax2=ax1.twinx()
    ax2.bar(index+0.3, human[result_index], color="r", width=0.3, label="Human inference")
    plt.xticks(index, label)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel("Avarage degree of human evaluation", fontsize=16)
    plt.ylim([3, 7])
    plt.legend(loc="lower right", bbox_to_anchor=(1.015, 1), fontsize=11.5)
    plt.savefig("result_4.eps", bbox_inches="tight", format="eps")
    plt.show()

