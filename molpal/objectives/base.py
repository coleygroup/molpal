from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, TypeVar

T = TypeVar('T')

class Objective(ABC):
    """An Objective is a class for calculating the objective function.

    An Objective DOES NOT indicate that values failed to be scored for any
    reason (e.g. with a specific value). If this happens, an objective should
    simply not return a label for that input.

    Attributes
    ----------
    c : int
        Externally, the objective is always maximized, so all values returned
        inside the Objective are first multiplied by c before being exposed to 
        a client of the Objective. If an objective is to be minimized, then c 
        is set to -1, otherwise it is set to 1.
    """
    def __init__(self, minimize: bool, **kwargs):
        self.c = -1 if minimize else 1

    def __call__(self, *args, **kwargs) -> Dict[T, Optional[float]]:
        self.calc(*args, **kwargs)

    @abstractmethod
    def calc(self,  xs: Collection[T], 
             *args, **kwargs) -> Dict[T, Optional[float]]:
        """Calculate the objective function for a collection of inputs

        Returns the value as a positive number. Does not include any inputs
        that failed to be labeled for any reason.
        """