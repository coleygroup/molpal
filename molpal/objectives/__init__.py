from typing import Iterable, Type

from molpal.objectives.base import Objective

# def objective(
#     *args, **kwargs
#     objectives: Iterable[str], objective_configs: Iterable[str],
#     minimize: Iterable[bool], path: str, verbose: int = 0,
#     **kwargs
# ) -> Objective:
#     return Objective(*args, **kwargs)
#     """Objective factory function"""
    
#     for o, config, mini in zip(objectives, objective_configs, minimize):
#         if o == 'docking':
#             from molpal.objectives.docking import DockingObjective
#             return DockingObjective(
#                 config, minimize=mini, path=path, verbose=verbose
#             )
#         if o == 'lookup':
#             from molpal.objectives.lookup import LookupObjective
#             return LookupObjective(config, **kwargs)
    
#     raise NotImplementedError(f'Unrecognized objective: "{objective}"')
