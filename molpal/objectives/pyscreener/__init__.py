from typing import Dict, List, Tuple

from molpal.objectives.pyscreener._version import __version__
from molpal.objectives.pyscreener.preprocessing import preprocess
from molpal.objectives.pyscreener.postprocessing import postprocess

def build_screener(mode, **kwargs) -> Tuple[Dict, List]:
    if mode == 'docking':
        from molpal.objectives.pyscreener import docking
        return docking.screener(**kwargs)
    if mode == 'md':
        from molpal.objectives.pyscreener import md
        raise NotImplementedError
    if mode == 'dft':
        from molpal.objectives.pyscreener import dft
        raise NotImplementedError
    
    raise ValueError(f'Unrecognized screening mode: "{mode}"')