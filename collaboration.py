import itertools
import numpy as np
from world import itempick
from lib import probutil


class Collaboration(object):
    def predictable_path_bak(self, world, reward_id, t=1):
        max_seq, max_prob = None, 0
        for a_seq in itertools.product([0, 1], repeat=t):
            prob = world.p_r__aseq(a_seq)[reward_id]
            if prob > max_prob:
                max_seq, max_prob = a_seq, prob
        return max_seq

    def predictable_path(self, world, reward_id, t=1):
        pos = world.agent
        a_seq = []
        prob = 1.0
        for i in range(t):
            p_r__a = world.p_r__a(pos)[:, reward_id]
            a = np.argmax(p_r__a)
            pos = pos + world.move_action.actions[a]
            pos = np.clip(pos, [0, 0], world.limit)
            a_seq.append(a)
            prob *= p_r__a[a]
        return a_seq, prob

    def best_path(self, world, reward_id, t=1):
        pos = world.agent
        a_seq = []
        prob = 1.0
        for i in range(t):
            p_a__r = world.p_a__r(pos)[:, reward_id]
            a = np.argmax(p_a__r)
            pos = pos + world.move_action.actions[a]
            pos = np.clip(pos, [0, 0], world.limit)
            a_seq.append(a)
            prob *= p_a__r[a]
        return a_seq, prob


if __name__ == '__main__':
    # size = (8, 13)
    # items = [((4, 1), 0), ((3, 3), 0), ((1, 5), 1), ((5, 6), 1), ((4, 9), 1), ((6, 9), 0)]
    # human, agent = (5, 11), (1, 1)
    #
    # size = (8, 8)
    # items = [((0, 0), 0), ((7, 1), 0), ((0, 5), 1), ((3, 2), 1)]
    # human, agent = [3, 7], [7, 6]

    size = (10, 10)
    # items = [((3, 7), 0), ((6, 5), 0), ((0, 5), 1), ((3, 2), 1)]
    items = [((3, 5), 0),
             ((4, 8), 0),
             ((6, 8), 0),
             ((5, 6), 1),
             ((9, 4), 2),
             ((6, 1), 0),
             ((4, 1), 0),
             ]
    # human, agent = [1, 6], [8, 2]
    human, agent = [9, 1], [1, 6]

    sample_world = itempick.ItemPickWorldItem(size, items, human, agent)

    print sample_world.p_r__a(1)
    p_reward_assign = probutil.make_prob_2d_dist(sample_world.r_assign_reward, 1)
    print np.dot(p_reward_assign.T, sample_world.p_assign__item.T).T

    top = sample_world.p_assign__item[1].argsort()[::-1][:10]
    # print sample_world.p_assign__item[1][top]
    # print sample_world.r_assign_reward[top]
    # for t in top:
    #     print sample_world.assign_action.all_action_seq[t].assign

    # solver = Collaboration()
    #
    # print solver.predictable_path(sample_world, 1, 3)
    # print solver.best_path(sample_world, 1, 3)

    sample_world.show_world()
