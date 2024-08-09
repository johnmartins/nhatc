import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem
from numpy.linalg import norm

lb = 1e-6
ub = 1e+6

coordinator = Coordinator()
coordinator.set_variables([
    ATCVariable('z3_1', 0, 0, False, [1], lb, ub),
    ATCVariable('z3_2', 1, 1, True, [0], lb, ub),
    ATCVariable('z4', 2, 0, False, [], lb, ub),
    ATCVariable('z5', 3, 0, False, [], lb, ub),
    ATCVariable('z6_1', 4, 0, False, [5], lb, ub),
    ATCVariable('z6_3', 5, 2, True, [4], lb, ub),
    ATCVariable('z7', 6, 0, False, [], lb, ub),
    ATCVariable('z8', 7, 1, False, [], lb, ub),
    ATCVariable('z9', 8, 1, False, [], lb, ub),
    ATCVariable('z10', 9, 1, False, [], lb, ub),
    ATCVariable('z11_2', 10, 1, False, [11], lb, ub),
    ATCVariable('z11_3', 11, 2, False, [10], lb, ub),
    ATCVariable('z12', 12, 2, False, [], lb, ub),
    ATCVariable('z13', 13, 2, False, [], lb, ub),
    ATCVariable('z14', 14, 2, False, [], lb, ub),
])


def sp1_objective(X):
    print('SP1')
    z3, z4, z5, z6, z7 = X[[0, 2, 3, 4, 6]]

    z1 = np.sqrt(np.pow(z3, 2) + np.pow(z4, -2) + np.pow(z5, 2))
    z2 = np.sqrt(np.pow(z5, 2) + np.pow(z6, 2) + np.pow(z7, 2))

    f = np.pow(z1, 2) + np.pow(z2, 2)
    y = []
    print(f)
    return f, y


@coordinator.prepare_constraint
def sp1_ieq_1(X):
    print('SP1 IEQ1')
    z3, z4, z5, z6, z7 = X[[0, 2, 3, 4, 6]]
    res = np.pow(z3, -2) + np.pow(z4, 2) - np.pow(z5, 2)

    print(res)
    return res


@coordinator.prepare_constraint
def sp1_ieq_2(X):
    print('SP1 IEQ2')
    z3, z4, z5, z6, z7 = X[[0, 2, 3, 4, 6]]
    res = np.pow(z5, 2) + np.pow(z6, -2) - np.pow(z7, 2)
    print(res)
    return res


def sp2_objective(X):
    print('SP2')
    z8, z9, z10, z11 = X[[7,8,9,10]]
    z3 = np.sqrt(np.pow(z8, 2) + np.pow(z9, -2)
                 + np.pow(z10, -2) + np.pow(z11, 2))

    f = 0
    y = z3

    return f, y


@coordinator.prepare_constraint
def sp2_ieq_1(X):
    print('SP2 IEQ1')
    z8, z9, z10, z11 = X[[7, 8, 9, 10]]
    return np.pow(z8, 2) + np.pow(z9, 2) - np.pow(z11, 2)


@coordinator.prepare_constraint
def sp2_ieq_2(X):
    print('SP2 IEQ2')
    z8, z9, z10, z11 = X[[7, 8, 9, 10]]
    return np.pow(z8, -2) + np.pow(z10, 2) - np.pow(z11, 2)


def sp3_objective(X):
    print('SP3')
    z11, z12, z13, z14 = X[[11,12,13,14]]
    z6_3 = np.sqrt(np.pow(z11, 2) + np.pow(z12, 2) + np.pow(z13, 2)
                   + np.pow(z14, 2))
    f = 0
    y = z6_3
    print(f'y = {y}')
    return f, y


@coordinator.prepare_constraint
def sp3_ieq_1(X):
    print('SP3 IEQ1')
    z11, z12, z13, z14 = X[[11, 12, 13, 14]]
    return np.pow(z11, 2) + np.pow(z12, -2) - np.pow(z13, 2)


@coordinator.prepare_constraint
def sp3_ieq_2(X):
    print('SP3 IEQ2')
    z11, z12, z13, z14 = X[[11, 12, 13, 14]]
    res = np.pow(z11, 2) + np.pow(z12, 2) - np.pow(z14, 2)
    print(X)
    print(res)
    return res


sp1 = SubProblem(0)
sp1.set_objective(sp1_objective)
sp1.set_ineqs([sp1_ieq_1, sp2_ieq_2])
sp2 = SubProblem(1)
sp2.set_objective(sp2_objective)
sp2.set_ineqs([sp2_ieq_1, sp2_ieq_2])
sp3 = SubProblem(2)
sp3.set_objective(sp3_objective)
sp3.set_ineqs([sp3_ieq_1, sp3_ieq_2])

x0 = np.array(np.random.uniform(low=lb, high=ub, size=15), dtype=float)
coordinator.set_subproblems([sp1, sp2, sp3])
X_star, F_star = coordinator.optimize(10000, x0, beta=1.2, gamma=0.25, convergence_threshold=1e-9, method='COBYLA')


print("Verification against objectives:")
print(f'Objective 1 F* = {sp1_objective(X_star)[0]}')
print(f'Objective 2 F* = {sp2_objective(X_star)[0]}')
print(f'Objective 3 F* = {sp3_objective(X_star)[0]}')

print(X_star)