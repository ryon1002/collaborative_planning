import itertools
import numpy as np


class Assign(object):
    def __init__(self, assign):
        self.assign = assign

    def startswith(self, other):
        return self.assign[0][:len(other.assign[0])] == other.assign[0] and \
               self.assign[1][:len(other.assign[1])] == other.assign[1]

    def add_raw_assign(self, assign):
        return Assign(((self.assign[0] + assign[0]), (self.assign[1] + assign[1])))

    def __add__(self, other):
        return Assign(((self.assign[0] + other.assign[0]), (self.assign[1] + other.assign[1])))


class AssignAction(object):
    def __init__(self, world):
        self.item_ids = np.arange(len(world.items))
        self.start_point = Assign(((self.item_ids[-1] + 1,), (self.item_ids[-1] + 2,)))

        self._all_action_seq = None

    @property
    def all_action_seq(self):
        if self._all_action_seq is None:
            self._all_action_seq = [self.start_point.add_raw_assign((order[:i], order[i:]))
                                    for order in itertools.permutations(self.item_ids)
                                    for i in range(len(self.item_ids) + 1)]
        return self._all_action_seq

    def get_all_action_seq(self, conditions=Assign(((), ()))):
        conditions = self.start_point + conditions
        for action_seq in self.all_action_seq:
            if action_seq.startswith(conditions):
                yield action_seq

    def get_all_condition(self, t=1):
        for i in itertools.permutations(self.item_ids, t):
            yield self.start_point.add_raw_assign(((), (i,)))

    def get_action_seq_index(self, conditions):
        return [n for n, a in enumerate(self.all_action_seq) if a.startswith(conditions)]


class MoveAction(object):
    def __init__(self):
        self.actions = np.array([[1, 0], [0, -1], [0, 1], [-1, 0]])
