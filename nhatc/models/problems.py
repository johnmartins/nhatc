import numpy as np

from typing import Optional, Callable
from plusminus import ArithmeticParser


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


class DynamicSubProblem(SubProblem):

    def __init__(self):
        super().__init__(SubProblem.TYPE_DYNAMIC)
        self.index = -1
        self.obj: str = "0"
        self.variables: dict[str, int] = {}
        self.ineqs: list[str] = []
        self.couplings: dict[str, str] = {}

        self.inequality_constraints = []
        self.equality_constraints = []

    def eval(self, X):
        parser = ArithmeticParser()

        for v in self.variables:
            eval_str = f'{v} = {X[self.variables[v]]}'
            try:
                parser.evaluate(eval_str)
            except ZeroDivisionError:
                eval_str = f'{v} = 9999999999'
                parser.evaluate(eval_str)

        y = []
        for c in self.couplings:
            eval_str = f'{c} = {self.couplings[c]}'
            try:
                parser.evaluate(eval_str)
                y.append(parser.evaluate(self.couplings[c]))
            except ZeroDivisionError:
                eval_str = f'{c} = 9999999999'
                parser.evaluate(eval_str)
                y.append(9999999999)

        return parser.evaluate(self.obj), np.array([y])

