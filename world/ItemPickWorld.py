import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

import matplotlib
matplotlib.use('Agg')

class ItemPickWorld():
    def __init__(self):
        # self.make_sample()
        return

        size = 3
        action = 2

        b_base_index = [(y, x) for y, x in
                        itertools.product(range(-size, size + 1), repeat=2) if
                        abs(x) + abs(y) <= size]
        # (abs(y) + abs(x)) % 2 == 0 and abs(x) + abs(y) <= 4 * size]

        base_index = []
        for y, x in b_base_index:
            if size > 2 and action > 1 and x == 1 and abs(y) == 1:
                continue
            # if x * y != 0 and abs(x) + abs(y) < 2 * size:
            #     continue
            # if x == 0 and abs(y) % 2 == 1:
            #     continue
            # if y == 0 and abs(x) % 2 == 1:
            #     continue
            base_index.append((y, x))

        base_index = np.array(base_index)
        pos = base_index * 4
        for p in pos:
            if p[1] >= 12 and action >= 3:
                p[1] += 6
            if p[1] >= 8 and action >= 2:
                p[1] += 4
            if p[1] >= 4:
                p[1] += 2

        pos[:, 0] += -1 * np.min(pos[:, 0]) + 1
        pos[:, 1] += -1 * np.min(pos[:, 1]) + 1
        self.map = np.zeros((np.max(pos[:, 0] + 2), np.max(pos[:, 1]) + 2))
        self.s = self.map.shape[0] * self.map.shape[1]

        self.items = [p for p in pos]
        # print pos
        #
        # pass

    def make_sample(self):
        # self.items = [((4, 4), (0, 0)), ((5, 5), (1, 1)), ((6, 6), (2, 2)), ((7, 7), (3, 3))]
        self.items = [
            ((5, 4), (0, 2)),
            ((4, 6), (1, 0)),
            ((2, 7), (2, 1)),
            ((13, 4), (1, 0)),
            ((15, 3), (2, 1)),
            ((14, 1), (3, 3)),
            ((7, 6), (3, 3)),
            ((12, 6), (2, 1)),
            ((14, 10), (3, 1)),
        ]
        self.map = np.zeros((17, 12))
        self.agent = (1, 1)
        self.s = self.map.shape[0] * self.map.shape[1]

    def show_world(self, path=(), save=False):
        color_map = {0: "blue", 1: "red", 2: "green", 3:"yellow"}
        for s in range(self.s):
            (y, x) = np.unravel_index(s, self.map.shape)
            plt.gca().add_patch(
                patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))

        for (y, x), (shape, color) in self.items:
            if shape == 0:
                plt.gca().add_patch(
                    patches.Rectangle((x + 0.15, y + 0.15), 0.7, 0.7, facecolor=color_map[color], edgecolor="k"))
            elif shape == 1:
                plt.gca().add_patch(
                    patches.Polygon(((x + 0.1, y + 0.1), (x + 0.9, y + 0.1), (x + 0.5, y + 0.5)), facecolor=color_map[color], edgecolor="k"))
            elif shape == 2:
                plt.gca().add_patch(
                    patches.Rectangle((x + 0.7, y + 0.1), 0.2, 0.4, facecolor=color_map[color], edgecolor="k"))
            elif shape == 3:
                plt.gca().add_patch(
                    patches.Circle((x + 0.5, y + 0.5), 0.25, facecolor=color_map[color], edgecolor="k"))

        y, x = self.agent
        plt.gca().add_patch(
            patches.RegularPolygon((x + 0.5, y + 0.5), 6, 0.4, facecolor="magenta", edgecolor="k"))
        # for s in path:
        #     (y, x) = np.unravel_index(s, self.map.shape)
        #     plt.gca().add_patch(
        #         patches.Circle((x + 0.5, y + 0.5), 0.1, facecolor="magenta", edgecolor="k"))
        for i in range(len(path) - 1):
            (y1, x1) = path[i]
            (y2, x2) = path[i + 1]
            # plt.Line2D(np.array([x1, x2]), np.array[(y1, y2)], color="red")
            plt.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], "m-", linewidth=3)
            # plt.gca().add_patch(
            #     patches.Circle((x + 0.5, y + 0.5), 0.1, facecolor="magenta", edgecolor="k"))

        plt.ylim(-0.02, self.map.shape[0])
        plt.xlim(0, self.map.shape[1]+ 0.02)
        plt.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off")
        plt.gca().set_aspect('equal', adjustable='box')
        if save:
            # plt.savefig("data/eval_" + str(self.eval_id) + "/stimuli.jpg", bbox_inches="tight")
            plt.savefig("data/eval_" + str(self.eval_id) + "/stimuli.eps", bbox_inches="tight", format="eps",
                        str=s)
        # plt.show()


if __name__ == '__main__':
    world = ItemPickWorld()
    import graph_world2
    graph = graph_world2.GraphWorld(world)
    exit()
    world.show_world([(1, 1), (1, 2), (1, 3), (1, 4),
                      (2, 4), (3, 4), (4, 4), (5, 4),
                      (6, 4), (7, 4), (8, 4), (9, 4),
                      (10, 4), (11, 4), (12, 4), (13, 4),
                      ])

