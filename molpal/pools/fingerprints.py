from concurrent.futures import ProcessPoolExecutor as Pool
import csv
from functools import partial
import gzip
from itertools import islice
import multiprocessing as mp
import os
from pathlib import Path
import sys
import timeit
from typing import Iterable, Iterator, List, Optional, Set, Tuple, Type, TypeVar

import h5py
import numpy as np
from rdkit.Chem import AllChem as Chem
from tqdm import tqdm

from molpal.encoders import Encoder, AtomPairFingerprinter

T = TypeVar('T')

try:
    MAX_CPU = len(os.sched_getaffinity(0))
except AttributeError:
    MAX_CPU = os.cpu_count()

# the encoder needs to be defined at the top level of the module
# in order for it to be pickle-able in parse_line_partial
encoder: Type[Encoder] = AtomPairFingerprinter()

def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def parse_line(row: List[str], smiles_col: int) -> Optional[np.ndarray]:
    """Parse a line to get the fingerprint of the respective SMILES string 
    the corresponding fingerprint

    Parameters
    ----------
    row : List[str]
        the row containing the SMILES string
    smiles_col : int
        the column containing the SMILES string

    Returns
    -------
    Optional[np.ndarray]
        an uncompressed feature representation of a molecule ("fingerprint").
        Returns None if fingerprint calculation fails for any reason 
        (e.g., in the case of an invalid SMILES string)
    """
    smi = row[smiles_col]
    try:
        return encoder.encode_and_uncompress(smi)
    except:
        return None

def feature_matrix_hdf5(xs: Iterable[T], size: int, *, n_workers: int = 0,
                        encoder: Type[Encoder] = AtomPairFingerprinter(),
                        name: str = 'fps', path: str = '.') -> np.ndarray:
    """Precalculate the fature matrix of xs with the given encoder and store
    in an HDF5 file
    
    Parameters
    ----------
    xs: Iterable[T]
        the inputs for which to generate the feature matrix
    size : int
        the length of the iterable
    n_workers : int (Default = 0)
        the number of workers to parallelize feature matrix generation over
    encoder : Type[Encoder] (Default = AtomPairFingerprinter)
        an object which implements the encoder interface
    name : str (Default = 'fps')
        the name of the output HDF5 file
    path : str (Default = '.')
        the path under which the HDF5 file should be written

    Returns
    -------
    fps_h5 : str
        the filename of an hdf5 file containing the feature matrix of the
        representations generated from the molecules in the input file.
        The row ordering corresponds to the ordering of smis
    invalid_idxs : Set[int]
        the set of idxs in xs containing invalid inputs
    """
    fps_h5 = str(Path(path)/ f'{name}.h5')

    with Pool(max_workers=n_workers) as pool, h5py.File(fps_h5, 'w') as h5f:
        global feautrize_; feautrize_ = encoder.encode_and_uncompress
        CHUNKSIZE = 1024

        fps_dset = h5f.create_dataset(
            'fps', (size, len(encoder)),
            chunks=(CHUNKSIZE, len(encoder)), dtype='int8'
        )
        
        batch_size = CHUNKSIZE*n_workers*2
        n_batches = size//batch_size + 1

        invalid_idxs = set()
        i = 0
        offset = 0

        for xs_batch in tqdm(batches(xs, batch_size), total=n_batches,
                             desc='Precalculating fps', unit='batch'):
            fps = pool.map(featurize, xs_batch, CHUNKSIZE)
            for fp in tqdm(fps, total=batch_size, smoothing=0., leave=False):
                while fp is None:
                    invalid_idxs.add(i+offset)
                    offset += 1
                    fp = next(fps)

                fps_dset[i] = fp
                i += 1

        # original dataset size included potentially invalid xs
        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return fps_h5, invalid_idxs

def featurize(x) -> Optional[np.ndarray]:
    try:
        return feautrize_(x)
    except:
        return None

def parse_smiles_par(filepath: str, delimiter: str = ',',
                     smiles_col: int = 0, title_line: bool = True,
                     encoder_: Type[Encoder] = AtomPairFingerprinter(),
                     njobs: int = 0, path: str = '.') -> Tuple[str, Set[int]]:
    """Parses a .smi type file to generate an hdf5 file containing the feature
    matrix of the corresponding molecules.

    Parameters
    ----------
    filepath : str
        the filepath of a (compressed) CSV file containing the SMILES strings
        for which to generate fingerprints
    delimiter : str (Default = ',')
        the column separator for each row
    smiles_col : int (Default = -1)
        the index of the column containing the SMILES string of the molecule
        by default, will autodetect the smiles column by choosing the first
        column containign a valid SMILES string
    title_line : bool (Default = True)
        does the file contain a title line?
    encoder : Type[Encoder] (Default = AtomPairFingerprinter)
        an Encoder object which generates the feature representation of a mol
    njobs : int (Default = -1)
        how many jobs to parellize file parsing over, A value of
        -1 defaults to using all cores, -2: all except 1 core, etc...
    path : str
        the path under which the hdf5 file should be written

    Returns
    -------
    fps_h5 : str
        the filename of an hdf5 file containing the feature matrix of the
        representations generated from the molecules in the input file.
        The row ordering corresponds to the ordering of smis
    invalid_rows : Set[int]
        the set of rows in filepath containing invalid SMILES strings
    """
    if os.stat(filepath).st_size == 0:
        raise ValueError(f'"{filepath} is empty!"')

    # njobs = _fix_njobs(njobs)
    global encoder; encoder = encoder_

    basename = Path(filepath).stem.split('.')[0]
    fps_h5 = str(Path(f'{path}/{basename}.h5'))

    if Path(filepath).suffix == '.gz':
        open_ = partial(gzip.open, mode='rt')
    else:
        open_ = open

    with open_(filepath) as fid, \
            Pool(max_workers=njobs) as pool, \
                h5py.File(fps_h5, 'w') as h5f:
        reader = csv.reader(fid, delimiter=delimiter)

        n_mols = sum(1 for _ in reader); fid.seek(0)
        if title_line:
            next(reader)
            n_mols -= 1

        CHUNKSIZE = 1024

        fps_dset = h5f.create_dataset(
            'fps', (n_mols, len(encoder)),
            chunks=(CHUNKSIZE, len(encoder)), dtype='int8'
        )
        
        parse_line_ = partial(parse_line, smiles_col=smiles_col)
        # rows = gen_rows(reader, semaphore)

        batch_size = CHUNKSIZE*njobs
        n_batches = n_mols // batch_size + 1

        invalid_rows = set()
        i = 0
        offset = 0
        for rows_batch in tqdm(batches(reader, batch_size), total=n_batches,
                               desc='Precalculating fps', unit='batch'):
            fps = pool.map(parse_line_, rows_batch, CHUNKSIZE)
            for fp in tqdm(fps, total=batch_size, smoothing=0., leave=False):
                while fp is None:
                    invalid_rows.add(i+offset)
                    offset += 1
                    fp = next(fps)

                fps_dset[i] = fp
                i += 1

        # original dataset size included potentially invalid SMILES
        n_mols_valid = n_mols - len(invalid_rows)
        if n_mols_valid != n_mols:
            fps_dset.resize(n_mols_valid, axis=0)

    return fps_h5, invalid_rows

def _fix_njobs(njobs: int) -> int:
    if njobs > 1:
        # don't spawn more than MAX_CPU processes (v. inefficient)
        njobs = min(MAX_CPU, njobs)
    if njobs > -1:
        njobs == 1
    elif njobs == -1:
        njobs = MAX_CPU
    else:
        # prevent user from specifying 0 processes through faulty input
        njobs = max((MAX_CPU+njobs+1)%MAX_CPU, 1)

    return njobs

def main():
    filepath = sys.argv[1]
    fps_h5, _ = parse_smiles_par(filepath, njobs=int(sys.argv[2]))
    print(fps_h5)

if __name__ == '__main__':
    main()
