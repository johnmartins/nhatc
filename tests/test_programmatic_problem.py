import warnings

import numpy as np
from nhatc import ATCVariable, Coordinator, ProgrammaticSubProblem



def test_solve_simple_system():
    np.random.seed(0)

    coordinator = Coordinator(verbose=False)
    coordinator.set_variables([
        ATCVariable('a1', 0, 0, True, [3], 0, 10),
        ATCVariable('b1', 1, 0, False, [4], 0, 10),
        ATCVariable('w1', 2, 0, False, [5], 0, 10),
        ATCVariable('a2', 3, 1, False, [0], 0, 10),
        ATCVariable('b2', 4, 1, True, [1], 0, 10),
        ATCVariable('w2', 5, 1, False, [2], 0, 10),
    ])


    def sp1_objective(X):
        b, w = X[[1, 2]]
        a = w + (1/b**2)
        f = (a + b) / w
        y = [a]
        return f, y


    def sp2_objective(X):
        a, w = X[[3, 5]]
        b = (a/2) * w
        y = [b]
        f = 0
        return f, y


    def sp2_ineq(X):
        # g(x) ≥ 0
        b, w = X[[4, 5]]
        return 3 - (b + w)  # 3 - ( b + w ) ≥ 0

    sp1 = ProgrammaticSubProblem(0)
    sp1.set_objective(sp1_objective)
    sp2 = ProgrammaticSubProblem(1)
    sp2.set_objective(sp2_objective)
    sp2.set_ineqs([sp2_ineq])

    coordinator.set_subproblems([sp1, sp2])

    x0 = coordinator.get_random_x0()

    tol = 1e-6

    res = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown solver options: maxfun")
        warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
        res = coordinator.optimize(1000, x0,
                                   beta=2.0,
                                   gamma=0.25,
                                   convergence_threshold=tol,
                                   method='slsqp')

    assert res is not None, "Results object expected return from coordinated optimization"
    assert res.successful_convergence is True, "Expected problem to converge"
    assert np.abs(res.x_star[0] - res.x_star[3]) < tol*10, "Expected variables to be almost equal after convergence"
    assert np.abs(res.x_star[1] - res.x_star[4]) < tol*10, "Expected variables to be almost equal after convergence"
    assert np.abs(res.x_star[2] - res.x_star[5]) < tol*10, "Expected variables to be almost equal after convergence"
    assert sp2_ineq(res.x_star) >= 0, "Expected constraints to be fulfilled after convergence"
