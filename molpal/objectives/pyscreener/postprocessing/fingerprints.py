"""This module contains functions for generating the feature matrix of a set of
molecules located either in a sequence of SMILES strings or in a file"""

import csv
from functools import partial
import gzip
import os
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import h5py
import numpy as np
from rdkit.Chem import AllChem as Chem
from tqdm import tqdm

def smi_to_fp(smi: str, radius: int = 2,
              length: int = 2048) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    return np.array(Chem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=length, useChirality=True)
    )

def gen_fps_h5(smis: Iterable[str], n_mols: int,
               path: str = '.', name: str = 'fps',
               radius: int = 2, length: int = 2048,
               distributed: bool = True,
               n_workers: int = -1, **kwargs) -> Tuple[str, Set[int]]:
    """Generate an hdf5 file containing the feature matrix of the list of
    SMILES strings

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings from which to generate fingerprints
    n_mols : Optional[int]
        the length of the iterable
    path : str (Default = '.')
        the path under which the H5 file should be written
    name : str (Default = 'fps')
        the name of the output H5 file
    radius : int (Default = 2)
        the radius of the fingerprints
    length : int (Default = 2048)
        the length of the fingerprints
    distributed : bool (Default = True)
        whether to parallelize fingerprint calculation over a distributed
        computing setup
    n_workers : int (Default = -1)
        how many jobs to parellize file parsing over.
        A value of -1 defaults to using all cores
    **kwargs
        additional and unused keyword arguments

    Returns
    -------
    fps_h5 : str
        the filename of the hdf5 file containing the feature matrix of the
        representations generated from the input SMILES strings
    invalid_idxs : Set[int]
        the set of indexes in the iterable containing invalid SMILES strings
    """
    if distributed:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor as Pool

        n_workers = MPI.COMM_WORLD.size
    else:
        from concurrent.futures import ProcessPoolExecutor as Pool
        if n_workers == -1:
            try:
                n_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                n_workers = os.cpu_count()

    fps_h5 = f'{path}/{name}.h5'
    compression = None

    with Pool(max_workers=n_workers) as pool, h5py.File(fps_h5, 'w') as h5f:
        CHUNKSIZE = min(1024, n_mols)
        fps_dset = h5f.create_dataset(
            'fps', (n_mols, length), compression=compression,
            chunks=(CHUNKSIZE, length), dtype='int8'
        )
        smi_to_fp_ = partial(smi_to_fp, radius=radius, length=length)

        invalid_idxs = set()
        offset = 0

        fps = pool.map(smi_to_fp_, smis, chunksize=CHUNKSIZE)
        for i, fp in tqdm(enumerate(fps), total=n_mols,
                          desc='Calculating fingerprints', unit='fp'):
            while fp is None:
                invalid_idxs.add(i+offset)
                offset += 1
                fp = next(fps)
            
            fps_dset[i] = fp

        n_mols_valid = n_mols - len(invalid_idxs)
        if n_mols_valid != n_mols:
            fps_dset.resize(n_mols_valid, axis=0)

    return fps_h5, invalid_idxs
