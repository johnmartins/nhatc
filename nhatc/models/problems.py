import numpy as np

from typing import Optional, Callable
import cexprtk


class SubProblem:
    TYPE_PROGRAMMATIC = 'programmatic'
    TYPE_DYNAMIC = 'dynamic'

    def __init__(self, type: str):
        self.index = -1
        self.type = type

        if self.type not in [SubProblem.TYPE_DYNAMIC, SubProblem.TYPE_PROGRAMMATIC]:
            raise ValueError(f'Unknown subproblem type {self.type}')


class ProgrammaticSubProblem(SubProblem):
    def __init__(self, index: int):
        super().__init__(SubProblem.TYPE_PROGRAMMATIC)
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

    def eval(self, X):
        return self.objective_function(X)

    def get_ineqs(self):
        return self.inequality_constraints

    def get_eqs(self):
        return self.equality_constraints


class DynamicSubProblem(SubProblem):

    def __init__(self):
        super().__init__(SubProblem.TYPE_DYNAMIC)
        self.index = -1
        self.obj: str = "0"
        self.variables: dict[str, int] = {}
        self.ineqs: list[str] = []
        self.couplings: dict[str, str] = {}

        self.inequality_constraints: list[str] = []
        self.equality_constraints: list[str] = []

        # Runtime vars
        self.symbol_table = cexprtk.Symbol_Table({}, {}, add_constants=True)

    def get_ineqs(self):
        return self.inequality_constraints

    def get_eqs(self):
        return self.equality_constraints

    def eval(self, X):

        # Set variables, build initial symbol table
        for v in self.variables:
            self.symbol_table.variables[v] = X[self.variables[v]]

        # Calculate coupling variables
        y = []
        for c in self.couplings:
            expr = cexprtk.Expression(self.couplings[c], self.symbol_table)
            value = expr()
            y.append(value)
            self.symbol_table.variables[c] = value

        # Calculate objective
        obj_expr = cexprtk.Expression(self.obj, self.symbol_table)
        return obj_expr(), np.array([y])
