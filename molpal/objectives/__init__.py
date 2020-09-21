from typing import Type

from molpal.objectives.base import Objective

def objective(objective, **kwargs) -> Type[Objective]:
    """Objective factory function"""
    if objective == 'docking':
        from .docking import DockingObjective
        return DockingObjective(**kwargs)
    if objective == 'lookup':
        from .lookup import LookupObjective
        return LookupObjective(**kwargs)
    # try:
    #     return {
    #         # 'docking': DockingObjective,
    #         'lookup': LookupObjective
    #     }[objective](**kwargs)
    
    raise NotImplementedError(f'Unrecognized objective: "{objective}"')
