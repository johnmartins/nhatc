from typing import Callable
from numpy import inf
from numpy.linalg import norm
import numpy as np


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
        self.scaling_vector = np.array([])      # s
        self.linear_weights = np.array([])      # v
        self.quadratic_weights = np.array([])   # w

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

    def optimize(self, i_max_outerloop: 10, initial_targets):
        """

        :param initial_targets: Initial guess for reasonable design
        :param i_max_outerloop: Maximum iterations of outer loop
        :return:
        """
        # Initial targets and inconsistencies
        X = initial_targets
        q_current = np.zeros(self.n_q)

        convergence_threshold = 0.1
        max_iterations = i_max_outerloop
        iteration = 0

        while iteration < max_iterations:
            print(f"Outer iteration {iteration}")

            for sub_problem in self.subproblems:
                pass

            # Update scaled inconsistency vector q (NOT c)
            q_previous = np.copy(q_current)
            q_current = self.get_updated_inconsistency_vector(X)
            self.update_weights(q_current, q_previous)

            epsilon = norm(q_previous - q_current)
            if epsilon < convergence_threshold:
                print("Convergence achieved.")
                break

            iteration += 1

