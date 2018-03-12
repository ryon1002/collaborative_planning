import itertools
import pulp

def make_map(item_num):
    problem = pulp.LpProblem('make_map', pulp.LpMaximize)
    var = pulp.LpVariable.dicts('VAR', (range(item_num), ["Y", "X"]), 0, 4, 'Integer')

    same_X = pulp.LpVariable.dicts('VAR', (range(item_num), range(item_num)), 0, 1, 'Integer')
    inv_X = pulp.LpVariable.dicts('VAR', (range(item_num), range(item_num)), 0, 1, 'Integer')
    same_Y = pulp.LpVariable.dicts('VAR', (range(item_num), range(item_num)), 0, 1, 'Integer')
    problem += pulp.lpSum([j for i in var.viewvalues() for j in i.viewvalues()])


    # print problem
    for i, j in itertools.combinations(range(4), 2):
        print type(var[j]["X"] * same_X[i][j])
        problem += same_X[i][j] * var[j]["X"] == 0


    status = problem.solve()
    print "Result"
    for i in range(item_num):
        print var[i]["Y"].value(), var[i]["X"].value()

if __name__ == '__main__':
    make_map(4)
