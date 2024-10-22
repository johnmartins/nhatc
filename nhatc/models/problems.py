import operator
import numpy as np

from typing import Optional, Callable
import cexprtk


class SubProblem:
    TYPE_PROGRAMMATIC = 'programmatic'
    TYPE_DYNAMIC = 'dynamic'

    def __init__(self, type: str):
        self.index = -1
        self.type = type
        self._coordinator = None

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
        self.symbol_order: dict[str, int] = {}

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

    def set_order_of_symbol(self, symbol, order):
        if symbol in self.symbol_order:
            raise ValueError(f'Symbol already in order table for sub system {self.index}')

        self.symbol_order[symbol] = order

    def get_symbol_order(self, symbol):
        """
        If there is a specified symbol order, return it.
        :param symbol:
        :return:
        """
        if symbol in self.symbol_order:
            return self.symbol_order[symbol]
        else:
            return -1

    def pre_compile(self, X, skip_if_initialized: bool = True, custom_functions: dict[str, callable] = {}):
        """
        Build all expressions and the symbol table
        :param X: Initial value of X
        :param skip_if_initialized: Do not run this function if it has been run before. Setting this to false enables running the function anyway.
        :return:
        """
        if skip_if_initialized is True and self.symbol_table_initialized:
            return

        if self._coordinator.verbose:
            print(f"Precompiling expressions for SubProblem {self.index}")

        # Set variables, build initial symbol table
        self.set_custom_functions(custom_functions)
        self.update_variables(X)

        # TODO: Loop through couplings AND intermediaries sorted by order of operation
        symbol_array = list(self.intermediates.keys()) + list(self.couplings)
        symbol_array.sort(key=lambda x: self.get_symbol_order(x)) #TODO: fix?!

        if self._coordinator.verbose:
            print(f"Expression order in SS {self.index}: {symbol_array}")

        for symbol in symbol_array:
            if symbol in self.intermediates:
                try:
                    self.inter_expr[symbol] = cexprtk.Expression(self.intermediates[symbol], self.symbol_table)
                    self.symbol_table.variables[symbol] = self.inter_expr[symbol]()
                except cexprtk.ParseException as err:
                    print(f'\nERROR: Failed to parse exception for symbol "{symbol}". \nReason: {repr(err)}')
                    print(f'Expression: \n{self.inter_expr[symbol]}\n')
                    raise err

            if symbol in self.couplings:
                try:
                    self.c_expr[symbol] = cexprtk.Expression(self.couplings[symbol], self.symbol_table)
                    self.symbol_table.variables[symbol] = self.c_expr[symbol]()
                except cexprtk.ParseException as err:
                    print(f'\nERROR: Failed to parse exception for symbol "{symbol}". \nReason: {repr(err)}')
                    print(f'Expression: \n{self.couplings[symbol]}\n')
                    raise err

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

    def set_custom_functions(self, custom_functions: dict[str, callable]):
        """
        Define any custom functions in the symbol table
        :param custom_functions:
        :return:
        """
        for key in custom_functions:
            self.symbol_table.functions[key] = custom_functions[key]

            if self._coordinator.verbose:
                print(f'Registered custom function "{key}" in SubProblem {self.index}')

    def get_ineqs(self):
        return self.inequality_constraints

    def get_eqs(self):
        return self.equality_constraints

    def add_intermediate_variable(self, symbol, expression):
        self.intermediates[symbol] = expression

    def eval(self, X):
        # TODO: Resolve variables based on order, if any
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
