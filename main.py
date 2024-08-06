import numpy as np
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from nhatc.models import ATCVariable, Coordinator
from numpy.linalg import norm

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 40)

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

coordinator.optimize(10, np.array([1,2,3,4,5,6,7,8,9,10]))