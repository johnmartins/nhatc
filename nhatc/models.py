from typing import Callable, Optional

from numpy import inf
from numpy.linalg import norm
import numpy as np

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


class Coordinator:

    def __init__(self):
        self.variables: list[ATCVariable] = []     # Array of variables
        self.subproblems = []
        self.beta = 2.2
        self.gamma = 0.25

        # Runtime variables
        self.X = np.array([], dtype=float)
        self.XD_indices = []
        self.XC_indices = []
        self.n_vars = 0   # Number of variables
        self.n_q = 0
        self.I = []
        self.I_prime = []
        self.scaling_vector = np.array([], dtype=float)      # s
        self.linear_weights = np.array([], dtype=float)      # v
        self.quadratic_weights = np.array([], dtype=float)   # w
        self.subsystem_in_evaluation = -1
        self.function_in_evaluation = None
        self.q_current = np.array([], dtype=float)
        self.xl_array = []
        self.xu_array = []
        self.var_name_map = {}
        self.F_star = []

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
        :param X: current values of X
        :return:
        """
        x_term = (self.X[self.I] - self.X[self.I_prime])
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
        self.X[self.XD_indices] = XD

        obj, y = self.function_in_evaluation(self.X)
        self.X[self.XC_indices] = y
        penalty_result = self.penalty_function()
        # print(f'subproblem {self.subsystem_in_evaluation} returns y = {y}')
        # print(self.X)
        # print(f'Penalty = {self.q_current}')

        return obj + penalty_result

    def run_subproblem_optimization(self, subproblem):
        print(f'Evaluating subproblem j = {self.subsystem_in_evaluation}')

        self.function_in_evaluation = subproblem.objective_function
        self.XD_indices = []
        self.XC_indices = []
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

        res = minimize(self.evaluate_subproblem, self.X[self.XD_indices],
                       method=None, # Let scipy decide, depending on presence of bounds and constraints
                       bounds=bounds, # Tuple of (min, max)
                       constraints=constraints) # List of dicts

        self.q_current = self.get_updated_inconsistency_vector()

        assert res.success, (f"Optimization process failed "
                             f"unexpectadly in subproblem {self.subsystem_in_evaluation}"
                             f"\nReason: {res.message}")

        self.F_star[self.subsystem_in_evaluation] = res.fun

    def optimize(self, i_max_outerloop: 10, initial_targets):
        """

        :param initial_targets: Initial guess for reasonable design
        :param i_max_outerloop: Maximum iterations of outer loop
        :return:
        """
        # Initial targets and inconsistencies
        self.X = initial_targets
        assert self.X.size == len(self.variables), "Initial guess x0 does not match specified variable vector size"
        self.q_current = np.zeros(self.n_q)

        convergence_threshold = 0.0001
        max_iterations = i_max_outerloop
        iteration = 0

        while iteration < max_iterations-1:
            print(f"Outer iteration {iteration}")
            q_previous = np.copy(self.q_current)

            for j, subproblem in enumerate(self.subproblems):
                self.subsystem_in_evaluation = j
                self.run_subproblem_optimization(subproblem)

            self.update_weights(self.q_current, q_previous)

            epsilon = norm(q_previous - self.q_current)
            if epsilon < convergence_threshold:
                with np.printoptions(precision=1, suppress=True):
                    print(f'{self.q_current}')
                    print(f'Epsilon = {epsilon}')
                    print(f"Convergence achieved after {iteration+1} iterations.")
                    print(f'X* = {self.X}')
                    print(f'F* = {self.F_star}')
                return

            iteration += 1

        print(f"Failed to converge after {iteration+1} iterations")

