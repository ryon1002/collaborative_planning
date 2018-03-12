import numpy as np
from world import itempick
from lib import probutil


class Collaboration(object):
    def predictable_path(self, world, t=1):
        ret = np.empty((len(world.items),))
        count = 0
        all_reward = world.r_assign_reward[:, 0]
        for i, c in enumerate(world.assign_action.get_all_condition(1)):
            index = world.action_type.get_action_seq_index(c)
            reward = all_reward[index]
            prob = probutil.make_prob_dist(reward, 1, 0)
            ret[count] = np.sum(-reward * prob)
            count += 1
        return ret


if __name__ == '__main__':
    # size = (8, 15)
    # items = [((4, 1), 1), ((2, 3), 0), ((2, 6), 0), ((5, 9), 1), ((3, 10), 1), ((6, 12), 0)]
    # human, agent = (4, 13), (1, 1)
    size = (8, 13)
    items = [((4, 1), 0), ((3, 3), 0), ((1, 5), 1), ((5, 6), 1), ((4, 9), 1), ((6, 9), 0)]
    human, agent = (5, 11), (1, 1)
    # size = (10, 15)
    # items = [((4, 1), 0), ((3, 3), 0), ((1, 5), 1), ((5, 6), 1), ((4, 9), 1), ((6, 9), 0)]
    # human, agent = (5, 11), (1, 1)
    # size = (8, 15)
    # items = [((1, 2), 0), ((6, 1), 0), ((2, 6), 0), ((5, 7), 0), ((1, 12), 0), ((5, 13), 0)]
    # human, agent = (1, 9), (3, 9)

    # sample_world = itempick.ItemPickWorldItem(size, items, human, agent)
    # sample_world = itempick.ItemPickWorldItem(size, items, human, agent, False)
    sample_world = itempick.ItemPickWorldMove(size, items, human, agent)

    import itertools
    # for a_seq in itertools.product([0, 1], repeat=3):
    for a_seq in itertools.product([0, 1], repeat=2):
        print a_seq, sample_world.p_r__aseq(a_seq)

    # print solver.predictable_path(sample_world, 1)
    # sample_world.show_world()
