from nhatc import ATCVariable, Coordinator, ProgrammaticSubProblem
import numpy as np


coordinator = Coordinator(verbose=True)
coordinator.set_variables([
    ATCVariable('a1', 0, 0, True, [4], 0, 10),
    ATCVariable('b1', 1, 0, False, [5], 0, 10),
    ATCVariable('u1', 2, 0, False, [6], 0, 10),
    ATCVariable('v1', 3, 0, False, [], 0, 10),
    ATCVariable('a2', 4, 1, False, [0], 0, 10),
    ATCVariable('b2', 5, 1, True, [1], 0, 10),
    ATCVariable('u2', 6, 1, False, [2], 0, 10),
    ATCVariable('w2', 7, 1, False, [], 0, 10)
])


def sp1_objective(X):
    u, v, b = X[[2, 3, 1]]
    a = (np.log(b) + np.log(u) + np.log(v))
    f = u + v + a + b
    y = [a]
    return f, y


def sp2_objective(X):
    u, w, a = X[[6, 7, 4]]
    b = u**-1 + w**-1 + a**1
    y = [b]
    f = 0
    return f, y


def sp2_ineq(X):
    # g(x) ≥ 0
    w, b = X[[7, 5]]
    return 10 - (w + b)


sp1 = ProgrammaticSubProblem(0)
sp1.set_objective(sp1_objective)
sp2 = ProgrammaticSubProblem(1)
sp2.set_objective(sp2_objective)
sp2.set_ineqs([sp2_ineq])

coordinator.set_subproblems([sp1, sp2])

x0 = coordinator.get_random_x0()

print(f'x0 = \t {x0}')
res = coordinator.optimize(100, x0,
                           beta=2.0,
                           gamma=0.25,
                           convergence_threshold=1e-9,
                           NI=60,
                           method='slsqp')

if res:
    if res.successful_convergence:
        print(f'Reached convergence')
    else:
        print(f'FAILED to reach convergence')

    print(f'Process time: {res.time} seconds')
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


