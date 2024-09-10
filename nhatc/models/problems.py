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
        self.intermediates: dict[str, str] = {}

        self.inequality_constraints: list[str] = []
        self.equality_constraints: list[str] = []

        ## Runtime vars
        self.symbol_table_initialized = False
        self.symbol_table = cexprtk.Symbol_Table({}, {}, add_constants=True)
        # Stored expressions
        self.obj_expr: Optional[cexprtk.Expression] = None
        self.c_expr: dict[str, cexprtk.Expression] = {}
        self.inter_expr: dict[str, cexprtk.Expression] = {}
        self.const_expr: dict[str, cexprtk.Expression] = {}

    def pre_compile(self, X, skip_if_initialized=True):
        """
        Build all expressions and the symbol table
        :param X: Initial value of X
        :param skip_if_initialized: Do not run this function if it has been run before. Setting this to false enables running the function anyway.
        :return:
        """
        if skip_if_initialized is True and self.symbol_table_initialized:
            return

        print("precompiling")

        # Set variables, build initial symbol table
        self.update_variables(X)

        for inter in self.intermediates:
            print(f"precompiling {inter} = {self.intermediates[inter]} to inters")
            self.inter_expr[inter] = cexprtk.Expression(self.intermediates[inter], self.symbol_table)
            self.symbol_table.variables[inter] = self.inter_expr[inter]()

        # Calculate coupling variables
        for c in self.couplings:
            self.c_expr[c] = cexprtk.Expression(self.couplings[c], self.symbol_table)
            self.symbol_table.variables[c] = self.c_expr[c]()

        # Precompile constraints
        for ieqc in self.inequality_constraints:
            self.const_expr[ieqc] = cexprtk.Expression(ieqc, self.symbol_table)
        for eqc in self.equality_constraints:
            self.const_expr[eqc] = cexprtk.Expression(eqc, self.symbol_table)

        self.obj_expr = cexprtk.Expression(self.obj, self.symbol_table)
        self.symbol_table_initialized = True

    def update_variables(self, X):
        """
        Update the symbol table with new values of the current variables.
        :param X: The current values of X
        :return:
        """
        for v in self.variables:
            self.symbol_table.variables[v] = X[self.variables[v]]

    def get_ineqs(self):
        return self.inequality_constraints

    def get_eqs(self):
        return self.equality_constraints

    def add_intermediate_variable(self, symbol, expression):
        self.intermediates[symbol] = expression

    def eval(self, X):
        # Set variables, build initial symbol table
        for v in self.variables:
            self.symbol_table.variables[v] = X[self.variables[v]]

        for inter in self.inter_expr:
            self.inter_expr[inter]()

        # Calculate coupling variables
        y = []
        for c in self.couplings:
            value = self.c_expr[c]()
            y.append(value)
            self.symbol_table.variables[c] = value

        # Calculate objective
        return self.obj_expr(), np.array([y])
