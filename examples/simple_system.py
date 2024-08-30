import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem
from numpy.linalg import norm


coordinator = Coordinator(verbose=True)
coordinator.set_variables([
    ATCVariable('a1', 0, 0, True, [3], 0, 10),
    ATCVariable('b1', 1, 0, False, [4], 0, 10),
    ATCVariable('w1', 2, 0, False, [5], 0, 10),
    ATCVariable('a2', 3, 1, False, [0], 0, 10),
    ATCVariable('b2', 4, 1, True, [1], 0, 10),
    ATCVariable('w2', 5, 1, False, [2], 0, 10),
])


def sp1_objective(X):
    a, b, w = X[[0, 1, 2]]
    a = w + (1/b**2)
    f = (a + b) / w
    y = [a]
    return f, y


def sp2_objective(X):
    a, b, w = X[[3, 4, 5]]
    b = (a/2) * w
    y = [b]
    f = 0
    return f, y


sp1 = SubProblem(0)
sp1.set_objective(sp1_objective)
sp2 = SubProblem(1)
sp2.set_objective(sp2_objective)

coordinator.set_subproblems([sp1, sp2])
F_star = [np.inf, 0]
attempt = 0
epsilon = 1
max_attempts = 1
res = None

while F_star[0] > 20 or F_star[0] < 0 or epsilon > 1e-8 or np.isnan(F_star[0]):
    attempt += 1

    if attempt > max_attempts:
        break

    x0 = coordinator.get_random_x0()
    print(f'x0 = \t {x0}')
    res = coordinator.optimize(100, x0,
                               beta=2.0,
                               gamma=0.25,
                               convergence_threshold=1e-9,
                               NI=60,
                               method='nelder-mead')

if res:
    if res.successful_convergence:
        print(f'Reached convergence after {attempt - 1} attempts')
    else:
        print(f'FAILED to reach convergence after {attempt - 1} attempts')

    print("Verification against objectives:")
    print(f'f* = {res.f_star[0]}')
    print(f'Epsilon = {res.epsilon} ')

    print('x*:')
    for i, x_i in enumerate(res.x_star):
        name = coordinator.variables[i].name
        lb = coordinator.variables[i].lb
        ub = coordinator.variables[i].ub

        print(f'{name}\t[{lb}; {ub}]\tvalue: {x_i}')
else:
    print('Catastrophic failure')


