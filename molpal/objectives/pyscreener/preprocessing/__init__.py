from typing import List

def preprocess(preprocessing_options: List[str], **kwargs):
    if 'none' in preprocessing_options:
        return kwargs

    if 'autobox' in preprocessing_options:
        from pyscreener.preprocessing.autobox import autobox
        kwargs['center'], kwargs['size'] = autobox(**kwargs)

    if 'pdbfix' in preprocessing_options:
        from pyscreener.preprocessing.pdbfix import pdbfix
        kwargs['receptors'] = [pdbfix(**kwargs)]
    
    if 'tautomers' in preprocessing_options:
        from pyscreener.preprocessing.tautomers import tautomers
        kwargs['ligands'] = tautomers(**kwargs)
        
    if 'filter' in preprocessing_options:
        from pyscreener.preprocessing.filter import filter_ligands
        kwargs['ligands'], kwargs['names'] = filter_ligands(**kwargs)

    return kwargs