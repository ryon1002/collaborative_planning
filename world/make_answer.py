import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

import matplotlib
matplotlib.use('Agg')

def make_answer(color, path, label=None):
    color_map = {0: "blue", 1: "red", 2: "green", 3:"yellow"}
    plt.clf()
    plt.gca().add_patch(
        patches.Rectangle((0.1, 0.1), 0.7, 0.7, facecolor=color_map[color[0]], edgecolor="k"))
    if len(color) > 2:
        plt.gca().add_patch(
            patches.Rectangle((0.60, 0.8), 0.2, 0.4, facecolor=color_map[color[2]], edgecolor="k"))
    if len(color) > 1:
        plt.gca().add_patch(
            patches.Polygon(((0.05, 0.8), (0.85, 0.8), (0.45, 1.3)), facecolor=color_map[color[1]], edgecolor="k"))
    if len(color) > 3:
        plt.gca().add_patch(
            patches.Circle((0.45, 0.45), 0.3, facecolor=color_map[color[3]], edgecolor="k"))

    plt.ylim(0, 1.5)
    plt.xlim(0, 1)
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')
    id = "_".join([str(c) for c in color])

    plt.text(-0.6, 0.65, "(" + label + ")", fontsize=48)
    # if label is not None:
    #     plt.ylabel("(a)")
    # plt.show()
    # exit()

    # plt.savefig(path, bbox_inches="tight")
    plt.savefig(path[:-4] + ".eps", bbox_inches="tight", format="eps")

if __name__ == '__main__':
    for i in range(1, 5):
        for c in itertools.product([0, 1, 2, 3], repeat=i):
            make_answer(c, "resource/sample.jpg")
        exit()
            # print c

# make_answer([0, 0, 1])
# make_answer([0, 1])
