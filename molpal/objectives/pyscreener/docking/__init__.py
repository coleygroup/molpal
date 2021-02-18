from typing import Dict

from molpal.objectives.pyscreener.docking.base import Screener

def screener(software, **kwargs):
    if software in ('vina', 'qvina', 'smina', 'psovina'):
        from molpal.objectives.pyscreener.docking.vina import Vina
        return Vina(software=software, **kwargs)
    
    if software in ('dock', 'ucsfdock', 'DOCK'):
        from molpal.objectives.pyscreener.docking.dock import DOCK
        return DOCK(**kwargs)
    
    raise ValueError(f'Unrecognized docking software: "{software}"')