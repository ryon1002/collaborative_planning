# -*- coding: utf-8 -*-
import openpyxl
import itertools

import ItemPickWorld

x_list = [chr(i) for i in range(65, 65 + 26)] + ["A" + chr(i) for i in range(65, 65 + 6)]
x_list = [(n, i) for n, i in enumerate(x_list)]
y_list = [(i, str(32 - i)) for i in range(32)]

colors = {1:-1, "FF0070C0": 0, "FFFF0000": 1, "FF00B050":2}
types = {u"☆": -1, u"■": 0, u"▲": 1, "I":2, u"●":3}

wb = openpyxl.load_workbook("stimulus.xlsx", read_only=True)

# ws, eval_id = wb["data_2_1_2"], 1
# ws, eval_id = wb["data_3_1_2"], 2
# ws, eval_id = wb["data_3_2_2"], 3
# ws, eval_id = wb["data_4_1_2"], 4
ws, eval_id = wb["data_4_2_2"], 5
# ws, eval_id = wb["data_4_3_2"], 6
# ws, eval_id = wb["data_2_1_3"], 7
# ws, eval_id = wb["data_3_2_3"], 8
# ws, eval_id = wb["data_4_3_3"], 9
# ws, eval_id = wb["data_2_1_2__"], 10
# ws, eval_id = wb["training"], 201
items = []
path = []
path_que = []
agent = (0, 0)
for (y, y_str), (x, x_str) in itertools.product(y_list, x_list):
    cell = ws[x_str + y_str]
    if cell.value is not None:
        if cell.value == u"☆":
            agent = (y, x)
            path.append((y, x))
        else:
            # print y, x, cell.value, cell.font.color.index
            items.append(((y, x), (types[cell.value], colors[cell.font.color.index])))
    if cell.fill is not None:
        if cell.fill.bgColor.index == 64:
            if cell.value != u"☆":
                path_que.append((y, x))
            # print "road", y, x
while len(path_que) > 0:
    y, x = path[-1][0], path[-1][1]
    for i in range(len(path_que)):
        if abs(path_que[i][0] - y) + abs(path_que[i][1] - x) == 1:
            path.append(path_que[i])
            path_que.pop(i)
            break
# print items, agent
x_s = [i[0][1] for i in items]
min_x, max_x = min(x_s), max(x_s)
min_x, size_x = min_x - 1, max_x - min_x + 3
y_s = [i[0][0] for i in items]
min_y, max_y = min(y_s), max(y_s)
min_y, size_y = min_y - 1, max_y - min_y + 3

agent = agent[0] - min_y, agent[1] - min_x
items = [((y - min_y, x - min_x), p) for (y, x), p in items]
path = [(y - min_y, x - min_x) for (y, x) in path]

# print size_x
# exit()
import numpy as np
np.set_printoptions(edgeitems=3200, linewidth=10000, precision=3)
world = ItemPickWorld.ItemPickWorld()
world.items = items
# world.map = np.zeros((32, 32))
world.map = np.zeros((size_y, size_x))
world.agent = agent
world.path = path
world.s = world.map.shape[0] * world.map.shape[1]
world.eval_id = eval_id

import pickle
pickle.dump(world, open("dump/sample_world_" + str(eval_id) + ".pkl", "w"))

# exit()

import graph_world2
graph_world2.GraphWorld(world, 0.5)

world.show_world(path, save=True)
# world.show_world([], save=True)
