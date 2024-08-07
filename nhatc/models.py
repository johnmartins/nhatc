from typing import Callable, Optional

from numpy import inf
from numpy.linalg import norm
import numpy as np

from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class SubProblem:
    def __init__(self, variable_indices: list[int],
                 constant_builders: list[Callable]):
        self.variable_indices: list[int] = variable_indices
        self.constant_builders = constant_builders
        self.C = np.zeros(len(constant_builders), dtype=float)
        self.objective_function: Optional[Callable] = None
        self.inequality_constraints: list[Callable] = []
        self.equality_constraints: list[Callable] = []
        self.constants: list[Callable] = []

    def set_objective(self, function: Callable):
        self.objective_function = function

    def update_constants(self, X):
        for i in range(0, len(self.constant_builders)):
            self.C[i] = self.constant_builders[i](X)


class ATCVariable:

    def __init__(self, index: int, subproblem_index: int,
                 coupled_variable: bool, links: list[int],
                 lb: -inf, ub: inf):
        """
        Variable definition based on Talgorn and Kokkolaras, 2017
        :param index: For identification
        :param subproblem_index: index of subproblem this variable belongs to
        :param coupled_variable: Is this variable sent to another subproblem?
        :param links: Index of variables that should, after convergence, have the same value as this variable
        :param lb: lower bound
        :param ub: upper bound
        """
        self.index = index
        self.subproblem_index = subproblem_index
        self.coupled_variable = coupled_variable
        self.links = links
        self.lb = lb
        self.ub = ub


class Coordinator:

    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 40)

    def __init__(self):
        self.variables: list[ATCVariable] = []     # Array of variables
        self.subproblems = []
        self.beta = 2.2
        self.gamma = 0.25

        # Runtime variables
        self.n_vars = 0   # Number of variables
        self.n_q = 0
        self.I = []
        self.I_prime = []
        self.scaling_vector = np.array([], dtype=float)      # s
        self.linear_weights = np.array([], dtype=float)      # v
        self.quadratic_weights = np.array([], dtype=float)   # w
        self.function_in_evaluation = None
        self.q_current = np.array([], dtype=float)
        self.xl_array = []
        self.xu_array = []

    def set_variables(self, variables: list[ATCVariable]):
        self.variables = variables
        self.n_vars = len(variables)

        self.I = []
        self.I_prime = []

        for var in self.variables:
            for linked_var_index in var.links:
                if var.index < linked_var_index:
                    self.I.append(var.index)
                    self.I_prime.append(linked_var_index)

        assert len(self.I) == len(self.I_prime)
        self.n_q = len(self.I)

        self.update_scaling_vector()
        self.linear_weights = np.zeros(self.n_q)
        self.quadratic_weights = np.ones(self.n_q)
        self.update_boundary_arrays()

    def update_boundary_arrays(self):
        self.xl_array = np.zeros(self.n_vars, dtype=float)
        self.xu_array = np.zeros(self.n_vars, dtype=float)

        for i, var in enumerate(self.variables):
            self.xl_array[i] = var.lb
            self.xu_array[i] = var.ub

    def update_scaling_vector(self):
        delta_array = []
        for index, var in enumerate(self.variables):
            delta = var.ub - var.lb
            if delta == 0 or delta == np.inf:
                delta = 1
            delta_array.append(delta)
        self.scaling_vector = np.array(delta_array)

    def get_updated_inconsistency_vector(self, X):
        """
        Returns the scaled inconsistency vector q
        :param X: current values of X
        :return:
        """
        x_term = (X[self.I] - X[self.I_prime])
        scale_term = (self.scaling_vector[self.I] + self.scaling_vector[self.I_prime])/2
        return x_term / scale_term

    def update_weights(self, q_current, q_previous):
        """
        Typically, 2 < beta < 3 and gamma = 0.25 (Tosseram, Etman, and Rooda, 2008)
        :param q_current: current scaled discrepancy vector
        :param q_previous: previous scaled discrepancy vector
        :return:
        """
        v = self.linear_weights
        w = self.quadratic_weights

        self.linear_weights = v + 2 * w * w * q_current

        for i in range(0, len(w)):
            if np.abs(q_current[i]) <= self.gamma * np.abs(q_previous[i]):
                self.quadratic_weights[i] = w[i]
            else:
                self.quadratic_weights[i] = self.beta * w[i]

    def penalty_function(self, scaled_inconsistency_vector):
        q = scaled_inconsistency_vector
        squared_eucl_norm = np.pow(norm(self.quadratic_weights * q), 2)
        return np.dot(self.linear_weights, q) + squared_eucl_norm

    def set_subproblems(self, subproblems: list[Callable]):
        self.subproblems = subproblems

    def evaluate_subproblem(self, X):
        function_result = self.function_in_evaluation(X)
        penalty_result = self.penalty_function(self.q_current)
        return function_result + penalty_result

    def run_subproblem_optimization(self, X, subproblem, xl, xu):

        # TODO: Update constants

        problem = FunctionalProblem(self.n_vars,
                                    [self.evaluate_subproblem],
                                    constr_ieq=[],
                                    constr_eq=[],
                                    xl=np.array([xl]),
                                    xu=np.array([xu]))

        res = minimize(problem, Coordinator.algorithm, Coordinator.termination,
                       seed=1,
                       save_history=True,
                       verbose=True)
        X_star = res.X
        F_star = res.F






    def optimize(self, i_max_outerloop: 10, initial_targets):
        """

        :param initial_targets: Initial guess for reasonable design
        :param i_max_outerloop: Maximum iterations of outer loop
        :return:
        """
        # Initial targets and inconsistencies
        X = initial_targets
        self.q_current = np.zeros(self.n_q)

        convergence_threshold = 0.1
        max_iterations = i_max_outerloop
        iteration = 0

        while iteration < max_iterations:
            print(f"Outer iteration {iteration}")

            for j, subproblem in enumerate(self.subproblems):
                self.run_subproblem_optimization(X, subproblem,
                                                 xl=self.xl_array[j],
                                                 xu=self.xu_array[j])

            # Update scaled inconsistency vector q (NOT c)
            q_previous = np.copy(self.q_current)
            self.q_current = self.get_updated_inconsistency_vector(X)
            self.update_weights(self.q_current, q_previous)

            epsilon = norm(q_previous - self.q_current)
            if epsilon < convergence_threshold:
                print("Convergence achieved.")
                break

            iteration += 1

