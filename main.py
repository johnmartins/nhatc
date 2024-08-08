import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem
from numpy.linalg import norm

"""
def sp1_analysis(get_variables):
    u1, v, a1, b1 = get_variables('u1', 'v1', 'b1')

    obj = u1 + v + a1 + b1
    y = np.log(u1) + np.log(v) + np.log(b1)

    return obj, y

def sp2_analysis():
    u2 = 0
    w = 0
    a2 = 0

    obj = 0
    y = np.pow(u2, -1) + np.pow(w, -1) + np.pow(a2, -1)

    return obj, y
"""


coordinator = Coordinator()
coordinator.set_variables([
    ATCVariable('u1', 0, 0, False, [4], 0, 10),
    ATCVariable('v', 1, 0, False, [], 0, 10),
    ATCVariable('a1', 2, 0, True, [6], 0, 10),
    ATCVariable('b1', 3, 0, False, [7], 0, 10),
    ATCVariable('u2', 4, 1, False, [0], 0, 10),
    ATCVariable('w', 5, 1, False, [], 0, 10),
    ATCVariable('a2', 6, 1, False, [2], 0, 10),
    ATCVariable('b2', 7, 1, True, [4], 0, 10)
])


def sp1_objective(X):
    u1 = X[0]
    v = X[1]
    b1 = X[3]
    a1 = np.log(X[0]) + np.log(X[1]) + np.log(X[3])

    f = u1 + v + a1 + b1
    y = a1
    return f, y


def sp1_ieq(X):
    # Prevent logarithmic runaway
    return 10 - np.log(X[0]) + np.log(X[1]) + np.log(X[2])


def sp2_objective(X):
    u2 = X[4]
    w = X[5]
    a2 = X[6]
    b2 = np.pow(u2, -1) + np.pow(w, -1) + np.pow(a2, -1)

    f = 0
    y = b2
    return f, y


def sp2_ieq(X,):
    b2 = np.pow(X[0], -1) + np.pow(X[1], -1) + np.pow(X[2], -1)
    w = X[1]
    return 10 + b2 - w


sp1 = SubProblem(0)
sp1.set_objective(sp1_objective)
sp1.set_ineqs([sp1_ieq])
sp2 = SubProblem(1)
sp2.set_objective(sp2_objective)
sp2.set_ineqs([sp2_ieq])


x0 = np.array([9.0] * 8)
coordinator.set_subproblems([sp1, sp2])
X_star, F_star = coordinator.optimize(100, x0, beta=2.5)


print("Verification against objectives:")
print(type(X_star))
print(f'Objective 1 F* = {sp1_objective(X_star)[0]}')
print(f'Objective 2 F* = {sp2_objective(X_star)[0]}')