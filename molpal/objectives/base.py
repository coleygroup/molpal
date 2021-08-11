from abc import ABC, abstractmethod
from typing import Collection, Dict, Iterable, List, Optional, TypeVar

T = TypeVar('T')

class Objective:
    def __init__(
        self, tasks: Iterable[str], task_configs: Iterable[str],
        minimize: Iterable[bool], path: str, verbose: int = 0,
        **kwargs
    ):
        self.tasks = []
        for task, config, m in zip(tasks, task_configs, minimize):
            if task == 'docking':
                from molpal.objectives.docking import DockingTask
                self.tasks.append(DockingTask(
                    config, minimize=m, path=path, verbose=verbose
                ))

            elif task == 'lookup':
                from molpal.objectives.lookup import LookupTask
                self.tasks.append(LookupTask(
                    config,  minimize=m, **kwargs
                ))
                
            else:
                raise NotImplementedError(f'Unrecognized task: "{task}"')

    def __len__(self):
        """the number of tasks in this objective"""
        return len(self.tasks)

    def __call__(self, *args, **kwargs) -> Dict[T, List[Optional[float]]]:
        self.calc(*args, **kwargs)

    def calc(self,  xs: Collection[T], 
             *args, **kwargs) -> Dict[T, List[Optional[float]]]:
        """Calculate the objective function for a collection of inputs"""
        scoress = [task(xs, *args, **kwargs) for task in self.tasks]
        return {
            key: [scores[key] for scores in scoress]
            for key in scoress[0]
        }

class Task(ABC):
    """A Task defines an interface calculating a single objective function
    value for a collection of inputs.

    A Task indicates values that failed to be scored for any reason with
    a value of None. Classes that implement the Task interface 
    should not utilize None in any other way.

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
        self.calc(*args, **kwargs)

    @abstractmethod
    def calc(self,  xs: Collection[T], 
             *args, **kwargs) -> Dict[T, Optional[float]]:
        """Calculate the objective function for a collection of inputs"""
