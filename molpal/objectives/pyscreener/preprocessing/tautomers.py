import os
from pathlib import Path
from typing import List

from .gypsum_dl.Start import prepare_molecules

def tautomers(ligands: List[str], path: str,
              num_workers: int = -1, ncpu: int = 1, distributed: bool = False,
              **kwargs) -> List[str]:
    output_folder = Path(path) / 'tautomers'
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    # not sure how to deal with gypsum-dl on a distributed system from within
    # my own code
    # if distributed:
    #     from mpi4py import MPI
    #     from mpi4py.futures import MPIPoolExecutor as Pool

    #     num_processors = ncpu
    # else:
    #     from concurrent.futures import ProcessPoolExecutor as Pool

    if num_workers == -1:
        try:
            num_processors = len(os.sched_getaffinity(0))
        except AttributeError:
            num_processors = os.cpu_count()
    else:
        num_processors = num_workers * ncpu

    for ligand_file in ligands:
        params = {
            'source': ligand_file,
            'output_folder': str(output_folder),
            'job_manager': 'multiprocessing',
            'num_processors': num_processors,
            'min_ph': 6.4,
            'max_ph': 8.4,
            'max_variants_per_compound': 3,
            'thoroughness': 5,
            'skip_optimize_geometry': False,
            'skip_alternate_ring_conformations': False,
            'skip_adding_hydrogen': False,
            'skip_making_tautomers': False,
            'skip_enumerate_chiral_mol': False,
            'skip_enumerate_double_bonds': False,
            'let_tautomers_change_chirality': False,
            'use_durrant_lab_filters': False
        }
        prepare_molecules(params)
    
    return list(map(str, output_folder.iterdir()))