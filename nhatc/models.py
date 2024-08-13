from typing import Callable, Optional

from numpy import inf
from numpy.linalg import norm
import numpy as np

from pymoo.problems.functional import FunctionalProblem

"""
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
"""

from scipy.optimize import minimize



class SubProblem:
    def __init__(self, index: int):
        self.index = index
        self.objective_function: Optional[Callable] = None
        self.inequality_constraints: list[Callable] = []
        self.equality_constraints: list[Callable] = []

    def set_objective(self, function: Callable):
        self.objective_function = function

    def set_ineqs(self, ineqs: list[Callable]):
        self.inequality_constraints = ineqs

    def set_eqs(self, eqs: list[Callable]):
        self.equality_constraints = eqs


class ATCVariable:

    def __init__(self, name: str, index: int, subproblem_index: int,
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
        self.name = name
        self.index = index
        self.subproblem_index = subproblem_index
        self.coupled_variable = coupled_variable
        self.links = links
        self.lb = lb
        self.ub = ub


class Result:
    def __init__(self, successful_convergence, epsilon, epsilon_diff, f_star, x_star):
        self.successful_convergence: bool = successful_convergence
        self.epsilon: float = epsilon
        self.epsilon_diff: float = epsilon_diff
        self.f_star = f_star
        self.x_star = x_star


class Coordinator:

    def __init__(self, verbose: bool = False):
        self.variables: list[ATCVariable] = []     # Array of variables
        self.subproblems: list[SubProblem] = []
        self.beta: float = 2.2
        self.gamma: float = 0.25
        self.verbose = verbose

        # Runtime variables
        self.X = np.array([], dtype=float)
        self.XD_indices: list[int] = []
        self.XC_indices: list[int] = []
        self.n_vars = 0   # Number of variables
        self.n_q = 0
        self.I = []
        self.I_prime = []
        self.scaling_vector = np.array([], dtype=float)      # s
        self.linear_weights = np.array([], dtype=float)      # v
        self.quadratic_weights = np.array([], dtype=float)   # w
        self.subproblem_in_evaluation: Optional[SubProblem] = None
        self.function_in_evaluation: Optional[Callable] = None
        self.q_current = np.array([], dtype=float)
        self.xl_array = []
        self.xu_array = []
        self.var_name_map = {}
        self.F_star = []
        self.inner_iteration = 0

    def get_random_x0(self):
        r = np.random.rand(len(self.variables))
        x0_list = []

        for i, var in enumerate(self.variables):
            x0_list.append(r[i] * (var.ub - var.lb) + var.lb)

        return np.array(x0_list)

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

        self.update_variable_name_map()
        self.update_scaling_vector()
        self.linear_weights = np.zeros(self.n_q)
        self.quadratic_weights = np.ones(self.n_q)
        self.update_boundary_arrays()

    def update_variable_name_map(self):
        self.var_name_map = {}
        for variable in self.variables:
            if variable.name in self.var_name_map:
                raise NameError(f'Variable names need to be unique. Found duplicate: {variable.name}')

            self.var_name_map[variable.name] = variable.index

        return self.var_name_map

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

    def get_updated_inconsistency_vector(self):
        """
        Returns the scaled inconsistency vector q
        :return:
        """
        x_term = (self.X[self.I] - self.X[self.I_prime])
        scale_term = (self.scaling_vector[self.I] + self.scaling_vector[self.I_prime])/2
        return x_term / scale_term

    def update_weights(self, q_previous):
        """
        :param q_previous: previous scaled discrepancy vector
        :return:
        """
        v = self.linear_weights
        w = self.quadratic_weights

        self.linear_weights = v + 2 * w * w * self.q_current

        for i in range(0, len(w)):
            if np.abs(self.q_current[i]) <= self.gamma * np.abs(q_previous[i]):
                self.quadratic_weights[i] = w[i]
            else:
                self.quadratic_weights[i] = self.beta * w[i]

    def penalty_function(self):
        q = self.get_updated_inconsistency_vector()
        squared_eucl_norm = np.pow(norm(self.quadratic_weights * q), 2)
        return np.dot(self.linear_weights, q) + squared_eucl_norm

    def set_subproblems(self, subproblems: list[SubProblem]):
        self.subproblems = subproblems
        self.F_star = [None] * len(self.subproblems)

    def get_variables(self, var_names: list[str]):
        vars = []
        for var_name in var_names:
            vars.append(self.X[self.var_name_map[var_name]])

        return vars

    def evaluate_subproblem(self, XD):
        # print(XD)
        self.X[self.XD_indices] = XD
        obj, y = self.function_in_evaluation(self.X)

        self.X[self.XC_indices] = y
        penalty_result = self.penalty_function()

        self.inner_iteration += 1

        total = obj + penalty_result

        # print(f'{obj} {penalty_result} {total}')

        return total

    def prepare_constraint(self, func):
        def wrapper(*args, **kwargs):
            return func(self.X)

        return wrapper

    def run_subproblem_optimization(self, subproblem, NI, method):
        self.function_in_evaluation = subproblem.objective_function
        self.XD_indices = []
        self.XC_indices = []

        if subproblem.index > len(self.subproblems) - 1:
            raise ValueError('Subproblem index higher than expected. Check subproblem indices.')

        bounds = []

        for var in self.variables:
            if var.subproblem_index == subproblem.index and var.coupled_variable:
                # Coupled variable
                self.XC_indices.append(var.index)
            elif var.subproblem_index == subproblem.index:
                # Design variable
                bounds.append([self.xl_array[var.index], self.xu_array[var.index]])
                self.XD_indices.append(var.index)
            else:
                continue

        constraints = []
        for c_ineq in subproblem.inequality_constraints:
            constraints.append({'type': 'ineq', 'fun': c_ineq})

        for c_eq in subproblem.equality_constraints:
            constraints.append({'type': 'eq', 'fun': c_eq})

        x0 = self.X[self.XD_indices]

        self.inner_iteration = 0

        # Print hessian?
        res = minimize(self.evaluate_subproblem, x0,
                       method=method, # Let scipy decide, depending on presence of bounds and constraints
                       bounds=bounds, # Tuple of (min, max)
                       constraints=constraints,
                       tol=1e-12,
                       options={
                           'maxiter': NI,
                           'maxfun': NI
                       }) # List of dicts

        if self.verbose:
            print(f"Minimized SP{subproblem.index} with {self.inner_iteration} inner iterations")

        self.q_current = self.get_updated_inconsistency_vector()
        self.X[self.XD_indices] = res.x
        self.F_star[self.subproblem_in_evaluation.index] = self.subproblem_in_evaluation.objective_function(self.X)

    def optimize(self, i_max_outerloop: 50, initial_targets,
                 beta=2.2,
                 gamma=0.25,
                 convergence_threshold=0.0001,
                 NI=60,
                 method: str = 'slsqp') -> Result:
        """
        :param NI: Max number of inner iterations (subproblem evals) per outer loop
        :param convergence_threshold: Difference between error between iterations before convergence
        :param gamma: gamma is typically set to about 0.25
        :param beta: Typically, 2 < beta < 3  (Tosseram, Etman, and Rooda, 2008)
        :param initial_targets: Initial guess for reasonable design
        :param i_max_outerloop: Maximum iterations of outer loop (NO)
        :return:
        """
        # Setup parameters
        self.beta = beta
        self.gamma = gamma
        convergence_threshold = convergence_threshold
        max_iterations = i_max_outerloop

        # Initial targets and inconsistencies
        self.X = initial_targets
        assert self.X.size == len(self.variables), "Initial guess x0 does not match specified variable vector size"
        self.q_current = np.zeros(self.n_q)

        iteration = 0
        epsilon = 0
        epsilon_diff = -1

        while iteration < max_iterations-1:
            q_previous = np.copy(self.q_current)

            for j, subproblem in enumerate(self.subproblems):
                self.subproblem_in_evaluation = subproblem
                self.run_subproblem_optimization(subproblem, NI, method)

            self.update_weights(q_previous)

            # Difference between this inner loop iteration and the previous
            epsilon_diff = norm(q_previous - self.q_current)
            # Total discrepancy between coupled vars in this loop iteration
            epsilon = norm(self.q_current)

            if epsilon_diff < convergence_threshold and epsilon < convergence_threshold:
                if self.verbose:
                    with np.printoptions(precision=3, suppress=True):
                        print(f'{self.q_current}')
                        print(f'Epsilon = {epsilon}')
                        print(f"Convergence achieved after {iteration+1} iterations.")
                        print(f'X* = {self.X}')
                        print(f'F* = {self.F_star}')
                return Result(True, epsilon, epsilon_diff, self.F_star, self.X)

            iteration += 1

        print(f"Failed to converge after {iteration+1} iterations")
        print(f'Epsilon = {epsilon}')
        return Result(False, epsilon, epsilon_diff, self.F_star, self.X)

