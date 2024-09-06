from numpy import inf


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

    def __str__(self):
        return f"{self.index}\t{self.name}\t{self.subproblem_index}\t{self.coupled_variable}\t{self.links}\t[{self.lb}, {self.ub}]"


