import yaml
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class GraphWorld(object):
    def __init__(self):
        self.beta = 0.5
        # data = yaml.load(open("graph_2.yaml", "r"))
        data = yaml.load(open("graph.yaml", "r"))
        self.graph = nx.DiGraph()
        for n, color in data["node"].viewitems():
            if color is None:
                self.graph.add_node(n)
            else:
                self.graph.add_node(n, color=color)
        for f, t, cost in data["edge"]:
            self.graph.add_edge(f, t, {'weight': int(cost)})

        self._calc_all_variables()

        a = (1, 3)
        print "result"
        print self.calc_top_down(a)
        print self.calc_bottom_up(a, self.p_p__a)
        print self.calc_bottom_up(a, self.p_p__a2)
    # def __init__(self, world):
    #     self.beta = 0.5
    #     items = {i:[] for i in range(4)}
    #     for (y, x), (type, color) in world.items:
    #          items[type].append(((y, x), color))
    #     self.graph = nx.DiGraph()
    #     self.graph.add_node(0, type=-1, y=world.agent[1], x=world.agent[0])
    #     node_idx = 1
    #     for type in range(4):
    #         for (y, x), color in items[type]:
    #             self.graph.add_node(node_idx, color=color, type=type, y=y, x=x)
    #             node_idx += 1
    #     nodes = self.graph.nodes(data=True)
    #     for i in range(len(nodes)):
    #         for j in range(i + 1, len(nodes)):
    #             node_i, node_j = nodes[i][1], nodes[j][1]
    #             if node_i["type"] + 1 == node_j["type"]:
    #                 distance = abs(node_i["x"] - node_j["x"]) + abs(node_i["y"] - node_j["y"])
    #                 self.graph.add_edge(nodes[i][0], nodes[j][0], {'weight': distance})
    #
    #     self._calc_all_variables()
    #
    #     a = (1, 3)
    #     print "result"
    #     print self.calc_top_down(a)
    #     print self.calc_bottom_up(a, self.p_p__a)
    #     print self.calc_bottom_up(a, self.p_p__a2)

    def calc_top_down(self, action):
        prob = []
        for r, path in self.paths.viewitems():
            p = 0
            for i in range(len(path)):
                if path[i][:len(action)]== action:
                    p += self.p_p__g[r][i]
            prob.append(p)
        return self._normalize(np.array(prob))

    def calc_bottom_up(self, action, p_g__a):
        prob = []
        for r in self.paths.viewkeys():
            path = self.a_p[action]
            p = 0
            for i in range(len(path)):
                if path[i] in self.paths[r]:
                    p += p_g__a[action][i]
            prob.append(p)
        return self._normalize(np.array(prob, dtype=np.float))

    def _calc_all_variables(self):
        self.colors = set([v["color"] for v in self.graph.node.viewvalues() if "color" in v])
        self.goals = {i: c for i, c in enumerate(self.colors)}
        self._calc_all_planning()
        self.actions = set([p[:2] for p_s in self.paths.viewvalues() for p in p_s])
        self.actions = [(a, self._calc_path_weight(a)) for a in sorted(self.actions)]
        print self.actions
        self.p_p__a, self.p_p__a2 = self.p_dist_p__a()
        # self.p__g = self.p_dist_p__g()

    def _calc_all_planning(self):
        self.paths = {}
        self.weights = {}
        self.p_p__g = {}
        for g_i, g in self.goals.viewitems():
            # self.paths[g_i] = {}
            # self.weights[g_i] = {}
            # self.p_g__c[g_i] = {}
            self.paths[g_i] = []
            self.weights[g_i] = []
            self.p_p__g[g_i] = []
            waypoint = set(
                [k for k, v in self.graph.node.viewitems() if "color" in v and v["color"] == g])
            for w_i, w in enumerate(waypoint):
                path = [tuple(p) for p in nx.all_simple_paths(self.graph, source=1, target=w)]
                weights = [self._calc_path_weight(p) for p in path]
                self.paths[g_i].extend(path)
                self.weights[g_i].extend(weights)
            self.p_p__g[g_i] = self._normalize(
                np.exp(-self.beta * np.array(self.weights[g_i], dtype=np.float)))
            # self.paths[g_i][w_i] = {j:p[1:] for j, p in enumerate(path)}
            # self.weights[g_i][w_i] = {j:we for j, we in enumerate(weights)}
            # self.p_g__c[g_i][w_i] = {j:p for j, p in enumerate(probs)}
        self.path_set = {r: set([p for p in path]) for r, path in self.paths.viewitems()}
        print self.paths
        print self.weights
        print self.p_p__g

    def p_dist_a__r_p(self):
        dist = {}
        for c in self.paths.viewkeys():
            dist[c] = {p[-1]: (p[1], 1.0) for p in self.paths[c]}
        return dist

    def p_dist_p__a(self):
        self.a_p = {}
        a__w = {}
        # a__w2 = {}
        for a, a_cost in self.actions:
            self.a_p[a] = []
            a__w[a] = []
            # a__w2[a] = []
            for g_i, p_s in enumerate(self.paths.viewvalues()):
                for p_i, p in enumerate(p_s):
                    if p[:len(a)] == a:
                        self.a_p[a].append(p)
                        a__w[a].append(self.weights[g_i][p_i])
                        # a__w2[a].append(self.weights[g_i][p_i] - a_cost)
        print self.a_p
        print a__w
        # print a__w2
        p__a = {a:self._normalize(np.exp(-self.beta * np.array(w, dtype=np.float))) for a, w in
                a__w.viewitems()}
        # p__a2 = {a:self._normalize(np.exp(-np.array(w, dtype=np.float))) for a, w in
        #         a__w2.viewitems()}
        # print p__a
        # print p__a2
        p__a2 = {a:np.ones_like(w) for a, w in a__w.viewitems()}
        return p__a, p__a2

    def _normalize(self, arr):
        return arr / np.sum(arr)

    def _calc_path_weight(self, path):
        return sum([self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)])


if __name__ == '__main__':
    a = GraphWorld()
