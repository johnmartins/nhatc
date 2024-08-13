import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem
from numpy.linalg import norm


coordinator = Coordinator(verbose=True)
coordinator.set_variables([
    ATCVariable('ww1', 0, 0, False, [6], 0, 1e+5),
    ATCVariable('wf1', 1, 0, False, [11], 0, 1e+5),
    ATCVariable('x02_1', 2, 1, False, [7], 0, 10),
    ATCVariable('x02_2', 3, 1, False, [8], 0, 10),
    ATCVariable('x2_1', 4, 1, False, [], 0, 10),
    ATCVariable('x2_2', 5, 1, False, [], 0, 10),
    ATCVariable('ww2', 6, 1, True, [0], 0, 1e+5),
    ATCVariable('x03_1', 7, 2, False, [2], 0, 10),
    ATCVariable('x03_2', 8, 2, False, [3], 0, 10),
    ATCVariable('x3_1', 9, 2, False, [], 0, 10),
    ATCVariable('x3_2', 10, 2, False, [], 0, 10),
    ATCVariable('wf3', 11, 2, True, [1], 0, 1e+5)
])


def sp1_objective(X):   # TODO: Fix because some vars are supposed to be vectors
    ww, wf = X[[0, 1]]
    f = 60000 + ww + wf
    y = []
    return f, y


def sp2_objective(X):
    x0_1, x0_2, x2_1, x2_2 = X[[2, 3, 4, 5]]
    x0 = np.array([x0_1, x0_2])
    x2 = np.array([x2_1, x2_2])
    ww = 4000 * (1 + norm(x0-1)**2) * (1 + norm(x2-1)**2)
    y = ww
    f = 0
    return f, y


def sp3_objective(X):
    x0_1, x0_2, x3_1, x3_2 = X[[7, 8, 9, 10]]
    x0 = np.array([x0_1, x0_2])
    x3 = np.array([x3_1, x3_2])
    xs = 10*(x0+x3)

    f_eh = lambda z: (-(z[1] + 47) * np.sin(np.sqrt(np.abs(z[1] + z[0]/2 + 47)))
                      - z[0] * np.sin(np.sqrt(np.abs(z[0] - (z[1] + 47)))))
    eh = f_eh(xs)
    omega = (1 + norm(x0-2)**2) * (1 + 0.001*norm(x3-2)**2) * (1 + 1000*np.abs(eh))
    dr = 0.025 + 0.004 * np.log10(omega)
    wf = 20000 + 380952 * dr + 9523809 * dr * dr

    f = 0
    y = wf

    return f, y


sp1 = SubProblem(0)
sp1.set_objective(sp1_objective)
sp2 = SubProblem(1)
sp2.set_objective(sp2_objective)
sp3 = SubProblem(2)
sp3.set_objective(sp3_objective)

coordinator.set_subproblems([sp1, sp2, sp3])
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
                               convergence_threshold=1e-10,
                               NI=60,
                               method='nelder-mead')

if res:
    if res.successful_convergence:
        print(f'Reached convergence after {attempt - 1} attempts')
    else:
        print(f'FAILED to reach convergence after {attempt - 1} attempts')

    print("Verification against objectives:")
    print(f'f* = {res.f_star}')
    print(f'Epsilon = {res.epsilon} ')

    print('x*:')
    for i, x_i in enumerate(res.x_star):
        name = coordinator.variables[i].name
        lb = coordinator.variables[i].lb
        ub = coordinator.variables[i].ub

        print(f'{name}\t[{lb}; {ub}]\tvalue: {x_i}')
else:
    print('Catastrophic failure')


