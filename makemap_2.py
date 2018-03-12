import itertools
import numpy as np


class MapMaker(object):
    def __init__(self, size, distance):
        self.size = size
        self.index = size[0] * size[1]
        self.pos = np.array(np.unravel_index(np.arange(self.index), self.size)).T
        self.distance = distance

    def make_map(self, item_num, start=0, end=0, base=[]):
        if end == 0:
            end = self.index
        if item_num == 0:
            yield []
        for i in range(start, end):
            for j in base:
                if self._distance(i, j) < self.distance:
                    break
            else:
                for l in self.make_map(item_num - 1, i + 1, end, base + [i]):
                    yield [i] + l

    def _distance(self, id1, id2):
        return np.sum(np.abs(self.pos[id1] - self.pos[id2]))

    def assign_prop(self, prop_nums):
        for assign in itertools.permutations(range(sum(prop_nums) + 2)):
            start = 0
            for nums in prop_nums:
                tmp_item = -1
                for item in assign[start:start + nums]:
                    if tmp_item > item:
                        break
                    tmp_item = item
                else:
                    start += nums
                    continue
                break
            else:
                yield assign


from world import itempick
import collaboration

if __name__ == '__main__':
    size = (8, 8)
    maker = MapMaker(size, 5)
    assigns = {a for a in maker.assign_prop([2, 2])}
    # items = [((4, 1), 0), ((3, 3), 0), ((1, 5), 1), ((5, 6), 1), ((4, 9), 1), ((6, 9), 0)]
    # human, agent = (5, 11), (1, 1)
    max_rate = 0
    for l in maker.make_map(6):
        for a in assigns:
            human, agent = maker.pos[l[a[-2]]], maker.pos[l[a[-1]]]
            items = []
            start = 0
            for i, num in enumerate([2, 2]):
                items.extend([(tuple(maker.pos[l[n]]), i) for n in a[start:start + num]])
                start += num
            sample_world = itempick.ItemPickWorldMove(size, items, human, agent)
            solver = collaboration.Collaboration()

            a_1, p_1 = solver.predictable_path(sample_world, 1, 3)
            a_2, p_2 = solver.best_path(sample_world, 1, 3)
            if a_1 != a_2:
                b_prob, s_prob = max(p_1, p_2), min(p_1, p_2)
                rate = b_prob / s_prob
                if rate > max_rate:
                    max_rate = rate
                    print items, human, agent
                    print a_1, a_2, p_1, p_2, rate
                    print
