from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, TypeVar, List
from molpal import objectives

T = TypeVar("T")


class Objective(ABC):
    """An Objective is a class for calculating the objective function.

    An Objective indicates values failed to be scored for any reason with a
    value of None. Classes that implement the objective interface should not
    utilize None in any other way.

    Attributes
    ----------
    c : int
        Externally, the objective is always maximized, so all values returned
        inside the Objective are first multiplied by c before being exposed to
        a client of the Objective. If an objective is to be minimized, then c
        is set to -1, otherwise it is set to 1.
    """

    def __init__(self, minimize: bool = False, **kwargs):
        self.c = -1 if minimize else 1

    def __call__(self, *args, **kwargs) -> Dict[T, Optional[float]]:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, xs: Collection[T], *args, **kwargs) -> Dict[T, Optional[float]]:
        """Calculate the objective function for a collection of inputs"""


class MultiObjective():
    """ A class for calculating the objective function for all objectives.
    Defines individual Objective classes and calls them
    Attributes:
    -----------
    objectives : List[Objective] with length = self.dim
    dim : number of objectives to be optimized

    Parameters:
    -----------
    objective_list: List of the objective types ('Lookup' or 'Docking')
        for each objective.
    objective_configs: List of paths to config file for each objective

    Note: len(objective_list) must equal len(objective_configs)!
    """

    def __init__(self, objective_list, obj_configs):
        if len(objective_list) != len(obj_configs): 
            print('ERROR: Need same number of objectives and obj configs')

        self.dim = len(objective_list)
        self.objectives = [objectives.objective(obj, config)
                           for obj, config in zip(objective_list, obj_configs)]

    def forward(self, smis: Collection[str]) -> Dict[str, List[float]]:
        scores = {}
        for smi in smis:
            scores[smi] = [self.objectives[i].c * self.objectives[i].data[smi]
                           if smi in self.objectives[i].data else None
                           for i in range(self.dim)]

        return scores

    def __call__(self, smis: Collection[str]) -> Dict[str, List[float]]:
        return self.forward(smis)
