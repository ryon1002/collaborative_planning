import unittest
import numpy as np
from world import itempick
import collaboration


class TestCollaboration(unittest.TestCase):
    def test_item_pick_1(self):
        size = (8, 15)
        items = [((4, 1), 1), ((2, 3), 0), ((2, 6), 0), ((5, 9), 1), ((3, 10), 1), ((6, 12), 0)]
        human, agent = (4, 13), (1, 1)
        world = itempick.ItemPickWorldItem(size, items, human, agent)

        test = world.p_r__a()
        expect = np.array([[0.39247995, 0.18633144, 0.42118861],
                           [0.15936728, 0.65741506, 0.18321765],
                           [0.71510567, 0.25453341, 0.03036092],
                           [0.10417247, 0.08611854, 0.80970899],
                           [0.04841822, 0.19305859, 0.75852319],
                           [0.00955636, 0.98094744, 0.0094962]])
        np.testing.assert_almost_equal(test, expect)

    def test_item_pick_2(self):
        size = (8, 13)
        items = [((4, 1), 0), ((3, 3), 0), ((1, 5), 1), ((5, 6), 1), ((4, 9), 1), ((6, 9), 0)]
        human, agent = (5, 11), (1, 1)
        world = itempick.ItemPickWorldMove(size, items, human, agent)

        test = world._p_item__move(1)
        expect = np.array([[0.27684366, 0.22337326, 0.03746672, 0.22337326],
                           [0.17142241, 0.13831337, 0.17142241, 0.13831337],
                           [0.03746672, 0.22337326, 0.27684366, 0.22337326],
                           [0.17142241, 0.13831337, 0.17142241, 0.13831337],
                           [0.17142241, 0.13831337, 0.17142241, 0.13831337],
                           [0.17142241, 0.13831337, 0.17142241, 0.13831337]])
        np.testing.assert_almost_equal(test, expect)

    def test_item_pick_3(self):
        size = (8, 15)
        items = [((1, 2), 0), ((6, 1), 0), ((2, 6), 0), ((5, 7), 0), ((1, 12), 0), ((5, 13), 0)]
        human, agent = (1, 9), (3, 9)
        world = itempick.ItemPickWorldItem(size, items, human, agent, False)
        solver = collaboration.Collaboration()

        expect = np.array(
            [19.25650808, 19.1124063, 16.01054232, 16.12215461, 17.03641691, 19.85173468])

        test = solver.predictable_path(world, 1)
        np.testing.assert_almost_equal(test, expect)


if __name__ == '__main__':
    unittest.main()
