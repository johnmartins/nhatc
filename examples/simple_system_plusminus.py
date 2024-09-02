import numpy as np

from nhatc import ATCVariable, Coordinator, DynamicSubProblem
from plusminus import ArithmeticParser


coordinator = Coordinator(verbose=True)
coordinator.set_variables([
    ATCVariable('a1', 0, 0, True, [3], 0, 10),
    ATCVariable('b1', 1, 0, False, [4], 0, 10),
    ATCVariable('w1', 2, 0, False, [5], 0, 10),
    ATCVariable('a2', 3, 1, False, [0], 0, 10),
    ATCVariable('b2', 4, 1, True, [1], 0, 10),
    ATCVariable('w2', 5, 1, False, [2], 0, 10),
])

spi_1 = DynamicSubProblem()
spi_1.index = 0
spi_1.obj = "(a + b) / w"
spi_1.variables = {'b': 1, 'w': 2}
spi_1.couplings = {'a': 'w + (1/(b^2))'}

spi_2 = DynamicSubProblem()
spi_2.index = 1
spi_2.obj = "0"
spi_2.variables = {'a': 3, 'w': 5}
spi_2.couplings = {'b': '(a/2) * w'}

spi_array = [spi_1, spi_2]

coordinator.set_subproblems(spi_array)
F_star = [np.inf, 0]
attempt = 0
epsilon = 1
max_attempts = 1

# x0 = np.array([6.76911903, 9.46969758, 1.13955465, 6.54515886, 5.03847838, 4.48557725])
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


