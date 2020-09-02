from typing import Type

from .base import Objective
# from .docking import DockingObjective
from .lookup import LookupObjective

def objective(objective, **kwargs) -> Type[Objective]:
    """Objective factory function"""
    try:
        return {
            # 'docking': DockingObjective,
            'lookup': LookupObjective
        }[objective](**kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized objective: "{objective}"')
