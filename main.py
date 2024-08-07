import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem
from numpy.linalg import norm


class SubProblem_1(SubProblem):

    def __init__(self):
        super().__init__(variable_indices=[0, 1, 3])


def sp1_objective(X):
    u1 = X[0]
    v = X[1]
    b1 = X[2]
    a1 = self.constants[0]
    return u1 + v + a1 + b1


sp1 = SubProblem([0, 1, 3], constant_builders=[
    lambda X: np.log(X[0]) + np.log(X[1]) + np.log(X[3])
])

sp1.set_objective(sp1_objective)

def subproblem_1(X):
    u1 = X[0]
    v = X[1]
    b1 = X[3]
    a1 = np.log(u1) + np.log(v) + np.log(b1)
    return u1 + v + a1 + b1


def subproblem_2(X):
    # wrt u2, w, a2
    X[7] = np.pow(X[4], -1) + np.pow(X[5], -1) + np.pow(X[6], -1)
    return 0

# TODO: Define inequalities. Consider making subproblems their own class,
#  containing their functions and constraint functions.
#  furthermore, add optimization step to coordinator, and update variables.

x0 = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

coordinator = Coordinator()
coordinator.set_variables([
    ATCVariable(0, 0, False, [4], 0, 10),
    ATCVariable(1, 0, False, [], 0, 10),
    ATCVariable(2, 0, True, [6], 0, 10),
    ATCVariable(3, 0, False, [7], 0, 10),
    ATCVariable(4, 1, False, [0], 0, 10),
    ATCVariable(5, 1, False, [], 0, 10),
    ATCVariable(6, 1, False, [2], 0, 10),
    ATCVariable(7, 1, True, [4], 0, 10)
])
coordinator.set_subproblems([subproblem_1, subproblem_2])
coordinator.optimize(10, x0)