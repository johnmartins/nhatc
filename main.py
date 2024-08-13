import numpy as np

from nhatc.models import ATCVariable, Coordinator, SubProblem


coordinator = Coordinator(verbose = True)
coordinator.set_variables([
    ATCVariable('u1', 0, 0, False, [4], 0.1, 10),
    ATCVariable('v', 1, 0, False, [], 0.1, 10),
    ATCVariable('a1', 2, 0, True, [6], 0.1, 10),
    ATCVariable('b1', 3, 0, False, [7], 0.1, 10),
    ATCVariable('u2', 4, 1, False, [0], 0.1, 10),
    ATCVariable('w', 5, 1, False, [], 0.1, 10),
    ATCVariable('a2', 6, 1, False, [2], 0.1, 10),
    ATCVariable('b2', 7, 1, True, [4], 0.1, 10)
])


def sp1_objective(X):
    u1, v, b1 = X[[0, 1, 3]]
    a1 = np.log(u1) + np.log(v) + np.log(b1)

    f = u1 + v + a1 + b1
    y = a1
    return f, y


@coordinator.prepare_constraint
def sp1_ieq(X):
    u1, v, b1 = X[[0, 1, 3]]
    # Prevent logarithmic runaway
    return (np.log(u1) + np.log(v) + np.log(b1)) - 10


def sp2_objective(X):
    u2, w, a2 = X[[4, 5, 6]]
    b2 = np.pow(u2, -1) + np.pow(w, -1) + np.pow(a2, -1)
    f = 0
    y = b2
    return f, y


@coordinator.prepare_constraint
def sp2_ieq(X):
    u2, w, a2 = X[[4, 5, 6]]
    b2 = np.pow(u2, -1) + np.pow(w, -1) + np.pow(a2, -1)
    return w - b2 + 10


sp1 = SubProblem(0)
sp1.set_objective(sp1_objective)
sp1.set_ineqs([sp1_ieq])
sp2 = SubProblem(1)
sp2.set_objective(sp2_objective)
sp2.set_ineqs([sp2_ieq])


coordinator.set_subproblems([sp1, sp2])
F_star = [np.inf, 0]
attempt = 0
epsilon = 1
max_attempts = 1

while F_star[0] > 20 or F_star[0] < 0 or epsilon > 1e-8 or np.isnan(F_star[0]):
    attempt += 1

    if attempt > max_attempts:
        print(f'Failed to reach target after {max_attempts} attempts')
        break

    x0 = np.array(np.random.uniform(low=0, high=10, size=8), dtype=float)
    x0 = np.array(np.ones(8) * 5)
    X_star, F_star, epsilon = coordinator.optimize(60, x0, beta=2.5, gamma=0.4,
                                          convergence_threshold=1e-12, NI=60)


print(f'Reached F_star target after {attempt - 1} attempts')
print("Verification against objectives:")
print(f'Objective 1 F* = {sp1_objective(X_star)[0]}')
print(f'Objective 2 F* = {sp2_objective(X_star)[0]}')
print(f'Epsilon = {epsilon} ')
print(f'IEQ_sp2 = {sp2_ieq(X_star)}')

print(F_star)
print(X_star)
