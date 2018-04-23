import networkx as nx
import numpy as np
from collections import defaultdict
import itertools
import make_answer


class GraphWorld(object):
    def __init__(self, world, beta, beta2=0.1):
        self.beta = beta
        self.beta2 = beta2
        items = {i: [] for i in range(4)}
        for (y, x), (type, color) in world.items:
            items[type].append(((y, x), color))
        self.graph = nx.DiGraph()
        self.graph.add_node(0, type=-1, y=world.agent[0], x=world.agent[1])

        self.properties = defaultdict(lambda: defaultdict(list))
        node_idx = 1
        for type in range(4):
            for (y, x), color in items[type]:
                self.graph.add_node(node_idx, color=color, type=type, y=y, x=x)
                self.properties[type][color].append(node_idx)
                node_idx += 1
        nodes = self.graph.nodes(data=True)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i][1], nodes[j][1]
                if node_i["type"] + 1 == node_j["type"]:
                    distance = abs(node_i["x"] - node_j["x"]) + abs(node_i["y"] - node_j["y"])
                    self.graph.add_edge(nodes[i][0], nodes[j][0], {'weight': distance})
        # print self.graph.edge

        base_action = [self.get_node_id(p) for p in world.path]
        self.base_action = tuple([a for a in base_action if a is not None])
        self.action_len = len(self.base_action)
        # print self.base_action
        # print self.graph.nodes(data=True)
        self._calc_all_variables()
        # print "--- answer ---"
        # self.check_action(self.base_action)

    def check_action(self, action):
        top_down = self.calc_top_down(action)
        # print top_down
        # print self.calc_bottom_up(action, self.p_p__a)
        # print self.calc_bottom_up2(action, self.p_p__a)
        # exit()
        # print self.p_p__a[action]
        # return top_down, self.calc_bottom_up2(action, self.p_p__a)
        return top_down, self.calc_bottom_up2(action, self.p_p__a), self.calc_bottom_up(action, self.p_p__a2)
        # print self.calc_bottom_up(action, self.p_p__a2)
        action_len = self._calc_path_weight(action)
        c_min_distance = []
        c_distance = []
        for i in range(len(self.goals)):
            tmp_c_min_distance = []
            tmp_c_distance = []
            for j, p in enumerate(self.paths[i]):
                if p[:len(action)] == action:
                    tmp_c_min_distance.append(self.weights[i][j])
                    tmp_c_distance.append(self.weights[i][j] - action_len)
            c_min_distance.append(min(tmp_c_min_distance) if len(tmp_c_min_distance) > 0 else -1)
            c_distance.append(np.sum(np.array(tmp_c_distance) < 10))
            # c_distance.append(np.array(tmp_c_distance) < 10)
        c_min_distance = np.array(c_min_distance)
        # print self._normalize(np.array(c_distance, dtype=np.float))
        # print np.array([min(self.weights[i]) for i in range(len(self.goals))])
        # print c_min_distance
        # print c_min_distance - action_len
        id = 1
        tmp = {8:"a", 9:"b", 10:"c", 11:"d"}
        for i, goal in sorted(self.goals.viewitems()):
            if i in tmp:
                make_answer.make_answer(goal, "check/answer_" + tmp[i] + ".jpg", tmp[i])
                # id += 1
            # make_answer.make_answer(goal, "check/answer_" + str(id) + ".jpg")
            # id += 1
            # pass
        # exit()

    def calc_top_down(self, action):
        prob = []
        for r, path in self.paths.viewitems():
            p = 0
            for i in range(len(path)):
                if path[i][:len(action)] == action:
                    p += self.p_p__g[r][i]
            prob.append(p)
        # print prob
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

    def calc_bottom_up2(self, action, p_g__a):
        prob = []
        # print self.p_p__g
        # print self.paths
        # print self.a_p
        for r in self.paths.viewkeys():
            # print r
            path = self.a_p[action]
            # print path
            # exit()
            # print path
            p = 0
            # print r, len(path), len(self.p_p__g[r])
            for i in range(len(path)):
                for j in range(len(self.paths[r])):
                    if path[i] == self.paths[r][j]:
                        # p += p_g__a[action][i]
                        p += p_g__a[action][i] * self.p_p__g[r][j]
            prob.append(p)
        return self._normalize(np.array(prob, dtype=np.float))

    def _calc_all_variables(self):
        self.goals = self._make_all_goals()
        # print "goals", self.goals
        self._calc_all_planning()
        self.actions = set([p[:self.action_len] for p_s in self.paths.viewvalues() for p in p_s])
        self.actions = [(a, self._calc_path_weight(a)) for a in sorted(self.actions)]
        # print "actions", self.actions
        self.p_p__a, self.p_p__a2 = self.p_dist_p__a()
        # print self.p_p__a
        # self.p__g = self.p_dist_p__g()

    def _make_all_goals(self):
        return {n: g for n, g in
                enumerate(itertools.product(*[v.keys() for v in self.properties.viewvalues()]))}

    def _calc_all_planning(self):
        self.paths = {}
        self.weights = {}
        self.p_p__g = {}
        self.p_p__g2 = {}
        for g_i, g in self.goals.viewitems():
            # self.paths[g_i] = {}
            # self.weights[g_i] = {}
            # self.p_g__c[g_i] = {}
            self.paths[g_i] = []
            self.weights[g_i] = []
            self.p_p__g[g_i] = []
            self.p_p__g2[g_i] = []
            for path in itertools.product(*[self.properties[t][i] for t, i in enumerate(g)]):
                path = [(0, ) + path]
                weights = [self._calc_path_weight(p) for p in path]
                self.paths[g_i].extend(path)
                self.weights[g_i].extend(weights)
            self.p_p__g[g_i] = self._normalize(
                np.exp(-self.beta * np.array(self.weights[g_i], dtype=np.float)))
            self.p_p__g2[g_i] = self._normalize(
                np.exp(-self.beta2 * np.array(self.weights[g_i], dtype=np.float)))
            # self.p_p__g3[g_i] = self._normalize(
            #     np.ones_like(self.weights[g_i]) / np.array(self.weights[g_i], dtype=np.float))

            # self.p_p__g[g_i] = self._normalize(
            #     np.exp(self.beta * (-np.array(self.weights[g_i], dtype=np.float)+30)))
            # self.paths[g_i][w_i] = {j:p[1:] for j, p in enumerate(path)}
            # self.weights[g_i][w_i] = {j:we for j, we in enumerate(weights)}
            # self.p_g__c[g_i][w_i] = {j:p for j, p in enumerate(probs)}
        self.path_set = {r: set([p for p in path]) for r, path in self.paths.viewitems()}
        # print "paths", self.paths
        # print "weights", self.weights
        # print "p_p__g", self.p_p__g

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
        # print self.a_p
        # print a__w
        # print a__w2
        # print a__w
        p__a = {a: self._normalize(np.exp(-self.beta2 * np.array(w, dtype=np.float))) for a, w in
                a__w.viewitems()}
        # p__a2 = {a: self._normalize(np.exp(np.ones_like(w) / np.array(w, dtype=np.float))) for a, w in
        #         a__w.viewitems()}
        # p__a2 = {a: self._normalize(np.exp(np.ones_like(w) / (np.array(w, dtype=np.float) +  1 - np.min(w)))) for a, w in
        #          a__w.viewitems()}
        p__a2 = {a: self._normalize(np.max(w) + 1 - (np.array(w, dtype=np.float))) for a, w in
                 a__w.viewitems()}
        # p__a3 = {a:np.exp(np.ones_like(w) * np.array(w, dtype=np.float)) for a, w in
        #          a__w.viewitems()}
        # p__a4 = {a:np.exp(np.array(w, dtype=np.float)) for a, w in
        #          a__w.viewitems()}
        # print p__a
        # print p__a2
        # print p__a3
        # print p__a4
        # p__a2 = {a:self._normalize(np.exp(-np.array(w, dtype=np.float))) for a, w in
        #         a__w2.viewitems()}
        # print p__a
        # print p__a2
        # p__a2 = {a: np.ones_like(w) for a, w in a__w.viewitems()}
        return p__a, p__a2

    def _normalize(self, arr):
        return arr / np.sum(arr)

    def _calc_path_weight(self, path):
        return sum([self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)])

    def get_node_id(self, (y, x)):
        for node in self.graph.nodes(data=True):
            if node[1]["x"] == x and node[1]["y"] == y:
                return node[0]
        return None

if __name__ == '__main__':
    import pickle

    # world = pickle.load(open("sample_world.pkl", "r"))
    # world = pickle.load(open("dump/sample_world_5.pkl", "r"))
    world.show_world()

    # g_world = GraphWorld(world)
