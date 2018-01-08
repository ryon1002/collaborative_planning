import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import squareform, pdist
import actiontype
import rewardtype
from lib import probutil


class ItemPickWorld(object):
    def __init__(self, size, items, human, agent, reward=True):
        self.map = np.zeros(size)

        self.items = [i[0] for i in items]
        self.human = np.array([human])
        self.agent = np.array([agent])
        self.dists = squareform(pdist(np.array(self.items + list(self.human) + list(self.agent)),
                                      metric="cityblock"))

        self.item_property = [i[1] for i in items]

        self.assign_action = actiontype.AssignAction(self)
        self.reward = rewardtype.RewardWeight(self, reward)

        self._r_assign_reward = None
        self._p_assign__reward = None
        self._p_item__reward = None

    @property
    def r_assign_reward(self):
        if self._r_assign_reward is None:
            self._r_assign_reward = np.array(
                [[self._ordered_assign_cost(a, r) for a in self.assign_action.all_action_seq]
                 for r in self.reward.get_all_rewards()]).T
        return self._r_assign_reward

    @property
    def p_assign__reward(self):
        if self._p_assign__reward is None:
            self._p_assign__reward = probutil.make_prob_2d_dist(self.r_assign_reward, 0)
        return self._p_assign__reward

    @property
    def p_item__reward(self):
        if self._p_item__reward is None:
            self._p_item__reward = np.array(
                [np.sum(self.p_assign__reward[self.assign_action.get_action_seq_index(c)], axis=0)
                 for c in self.assign_action.get_all_condition(1)])
            self._p_item_reward = probutil.make_prob_2d_dist(self._p_item__reward, 0)
        return self._p_item__reward

    def _ordered_assign_cost(self, assigns, weight):
        return min(self._single_ordered_assign_cost(assigns.assign[0], weight[0]),
                   self._single_ordered_assign_cost(assigns.assign[1], weight[1]))

    def _single_ordered_assign_cost(self, assign, weight):
        return -sum([self.dists[assign[i], assign[i + 1]] for i in range(len(assign) - 1)]) \
               - np.sum(weight[list(assign)])

    def show_world(self, path=()):
        color_map = {0: "lightblue", 1: "red"}

        for y, x in itertools.product(*[range(i) for i in self.map.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))
        for i, (y, x) in enumerate(self.items):
            plt.gca().add_patch(
                patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor=color_map[self.item_property[i]],
                               edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.human[0, 1] + 0.5, self.human[0, 0] + 0.5),
                                           0.4, facecolor="lightgreen", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.agent[0, 1] + 0.5, self.agent[0, 0] + 0.5),
                                           0.4, facecolor="pink", edgecolor="k"))
        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


class ItemPickWorldItem(ItemPickWorld):
    def __init__(self, size, items, human, agent, reward=True):
        super(ItemPickWorldItem, self).__init__(size, items, human, agent, reward)
        self.action_type = actiontype.AssignAction(self)

    def p_r__a(self, beta=1):  # P(r | a)
        return probutil.normalized_2d_array(self.p_item__reward, 1)


class ItemPickWorldMove(ItemPickWorld):
    def __init__(self, size, items, human, agent, reward=True):
        super(ItemPickWorldMove, self).__init__(size, items, human, agent, reward)
        self.action_type = actiontype.MoveAction()

    def _p_move__item(self, item, beta=1):
        pos = self.agent + self.action_type.actions
        q = -np.sum(np.abs(pos - np.array(self.items[item])), axis=1)
        return probutil.make_prob_dist(q, beta)

    def _p_item__move(self, beta=1):
        p_move__item = np.array([self._p_move__item(i, beta) for i in range(len(self.items))])
        return probutil.normalized_2d_array(p_move__item, 0)

    def p_r__a2(self, beta=1):  # P(r | a)
        p_item__move = self._p_item__move(beta)
        p_assign__r = probutil.make_prob_2d_dist(self.r_assign_reward, 0, beta)  # P(r | As)
        p_a__r = np.array([np.sum(p_assign__r[self.assign_action.get_action_seq_index(c)], axis=0)
                           for c in self.assign_action.get_all_condition(1)])
        p_r__item = probutil.normalized_2d_array(p_a__r, 1)
        return np.dot(p_item__move.T, p_r__item)
