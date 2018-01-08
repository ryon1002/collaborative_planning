import numpy as np


class RewardWeight(object):
    def __init__(self, world, multi_reward=True):
        if multi_reward:
            reward_list = [({0: 0, 1: 0}, {0: 0, 1: 0}),
                           ({0: 0, 1: 0}, {0: 0, 1: 10}),
                           ({0: 0, 1: 0}, {0: 10, 1: 0})]
        else:
            reward_list = [({0: 0, 1: 0}, {0: 0, 1: 0})]
        self.weight_list = np.array(
            [self._make_weight(r, world.item_property) for r in reward_list])

    def _make_weight(self, reward, item_property):
        return [[r[i] for i in item_property] + [0, 0] for r in reward]

    def get_all_rewards(self):
        for r in self.weight_list:
            yield r

    def size(self):
        return len(self.weight_list)
