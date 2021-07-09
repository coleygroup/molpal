from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple, TypeVar

import h5py
import numpy as np
import ray
from tqdm import tqdm

from molpal.featurizer import Featurizer, featurize, feature_matrix

T = TypeVar('T')

def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

# @ray.remote
# def _smis_to_fps(smis: Iterable[str], fingerprint: str = 'pair',
#                  radius: int = 2,
#                  length: int = 2048) -> List[Optional[np.ndarray]]:
#     fps = [featurize(smi, fingerprint, radius, length) for smi in smis]
#     return fps

# def smis_to_fps(smis: Iterable[str], fingerprint: str = 'pair',
#                 radius: int = 2,
#                 length: int = 2048) -> List[Optional[np.ndarray]]:
#     """
#     Caculate the Morgan fingerprint of each molecule in smis

#     Parameters
#     ----------
#     smis : Iterable[str]
#         the SMILES strings of the molecules
#     radius : int, default=2
#         the radius of the fingerprint
#     length : int, default=2048
#         the number of bits in the fingerprint

#     Returns
#     -------
#     List
#         a list of the corresponding morgan fingerprints in bit vector form
#     """
#     chunksize = int(ray.cluster_resources()['CPU'] * 64)
#     refs = [
#         _smis_to_fps.remote(smis_chunk, fingerprint, radius, length)
#         for smis_chunk in batches(smis, chunksize)
#     ]
#     fps_chunks = [
#         ray.get(r)
#         for r in tqdm(refs, desc='Calculating fingerprints',
#                       unit='chunk', leave=False)
#     ]
#     fps = list(chain(*fps_chunks))

#     return fps

def feature_matrix_hdf5(smis: Iterable[str], size: int, *,
                        featurizer: Featurizer = Featurizer(),
                        name: str = 'fps.h5',
                        path: str = '.') -> Tuple[str, Set[int]]:
    """Precalculate the fature matrix of xs with the given featurizer and store
    the matrix in an HDF5 file
    
    Parameters
    ----------
    xs: Iterable[T]
        the inputs for which to generate the feature matrix
    size : int
        the length of the iterable
    ncpu : int (Default = 0)
        the number of cores to parallelize feature matrix generation over
    featurizer : Featurizer, default=Featurizer()
        an object that encodes inputs from an identifier representation to
        a feature representation
    name : str (Default = 'fps.h5')
        the name of the output HDF5 file with or without the extension
    path : str (Default = '.')
        the path under which the HDF5 file should be written

    Returns
    -------
    fps_h5 : str
        the filename of an hdf5 file containing the feature matrix of the
        representations generated from the molecules in the input file.
        The row ordering corresponds to the ordering of smis
    invalid_idxs : Set[int]
        the set of indices in xs containing invalid inputs
    """
    fps_h5 = str((Path(path) / name).with_suffix('.h5'))

    # fingerprint = featurizer.fingerprint
    # radius = featurizer.radius
    # length = featurizer.length

    ncpu = int(ray.cluster_resources()['CPU'])
    with h5py.File(fps_h5, 'w') as h5f:
        CHUNKSIZE = 512

        fps_dset = h5f.create_dataset(
            'fps', (size, len(featurizer)),
            chunks=(CHUNKSIZE, len(featurizer)), dtype='int8'
        )
        
        batch_size = CHUNKSIZE * 2 * ncpu
        n_batches = size//batch_size + 1

        invalid_idxs = set()
        i = 0
        offset = 0

        for smis_batch in tqdm(batches(smis, batch_size), total=n_batches,
                             desc='Precalculating fps', unit='batch'):
            fps = feature_matrix(smis_batch, featurizer)
            for fp in tqdm(fps, total=batch_size, smoothing=0., leave=False):
                if fp is None:
                    invalid_idxs.add(i+offset)
                    offset += 1
                    continue
                    # fp = next(fps)

                fps_dset[i] = fp
                i += 1

        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return fps_h5, invalid_idxs
