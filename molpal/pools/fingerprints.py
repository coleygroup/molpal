from itertools import chain, islice
from pathlib import Path
from typing import Iterable, Iterator, List, Set, Tuple, TypeVar

import h5py
import numpy as np
import ray
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import cDataStructs
from tqdm import tqdm

from molpal.encoder import Featurizer, feature_matrix

T = TypeVar('T')

def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

@ray.remote
def _smis_to_fps(smis: Iterable[str],
                 radius: int = 2,
                 length: int = 2048) -> List[cDataStructs.ExplicitBitVect]:
    return [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), radius, length, useChirality=True
        ) for smi in smis
    ]

def smis_to_fps(smis: Iterable[str],
                radius: int = 2,
                length: int = 2048) -> List[cDataStructs.ExplicitBitVect]:
    """
    Caculate the Morgan fingerprint of each molecule in smis

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings of the molecules
    radius : int, default=2
        the radius of the fingerprint
    length : int, default=2048
        the number of bits in the fingerprint

    Returns
    -------
    List
        a list of the corresponding morgan fingerprints in bit vector form
    """
    chunksize = int(ray.cluster_resources()['CPU'] * 512)
    refs = [
        _smis_to_fps.remote(smis_chunk, radius, length)
        for smis_chunk in batches(smis, chunksize)
    ]
    fps_chunks = [
        ray.get(r)
        for r in tqdm(refs, desc='Building mols', unit='chunk')
    ]
    fps = list(chain(*fps_chunks))

    return fps

@ray.remote
def _neighbors_batch(
        idxs: Iterable[int],
        fps: List[cDataStructs.ExplicitBitVect],
        threshold: float
    ) -> np.ndarray:
    """Get the k nearest neighbors in fps for each fingerprint fp[i] in idxs

    Parameters
    ----------
    idxs : Iterable[int]
        the indices for which to calculate the nearest neighbors
    fps : List[cDataStructs.ExplicitBitVect]
        the chemical fingerprints in which to search for the nearest neighbors
    threshold: float
        the distance threshold below which a molecule is considered a neighbor. 
        NOTE: distances are in the range [0, 1]. Using a threshold value less 
        than 0 will result in no molecules being considered neighbors. Using a 
        threshold value equal to or greater than 1 will result in all molecules 
        being considered neighbors

    Returns
    -------
    np.ndarray
        the indices of the k nearest neighbors for each index i in idxs in the
        original ordering of idxs
    """
    def nns(i):
        X_d = 1. - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        return np.where(X_d < threshold)[0]

    return [nns(i) for i in idxs]

def neighbors(smis: Iterable[str], threshold: float) -> List[np.ndarray]:
    """For each molecule in in the list of SMILES strings, get the indices of all other molecules that are below a distance threshold

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings of the molecules
    threshold: float
        the distance threshold below which a molecule is considered a neighbor.
        NOTE: molecules are considered their own neighbor because
            dist(smis[i], smis[i]) == 0
        NOTE: distances are in the range [0, 1]. Using a threshold value less 
            than or equal to 0 will result in only the parent molecule being 
            returned as a neighbor. Using a threshold value equal to or greater 
            than 1 will result in all molecules being considered neighbors

    Returns
    -------
    neighbor_idxs : List[np.ndarray]
        the indices of the neighbors for the molecule represented by smis[i].
        This list has the same ordering as the input list smis. NOTE: includes 
        the parent molecule (i.e., neighbor_idxs[i] will always contain i)
    """
    if threshold <= 0:
        return [[i] for i in range(len(smis))]
    if threshold >= 1:
        idxs = list(range(len(smis)))
        return [idxs for _ in smis]

    fps = smis_to_fps(smis)
    idxs = range(len(fps))
    fps = ray.put(fps)

    chunksize = int(ray.cluster_resources()['CPU'] * 64)
    refs = [
        _neighbors_batch.remote(idxs_chunk, fps, threshold)
        for idxs_chunk in batches(idxs, chunksize)
    ]
    neighbor_idxs_chunks = [
        ray.get(r)
        for r in tqdm(refs, desc='Calculating neighbors', unit='chunk')
    ]
    
    return list(chain(*neighbor_idxs_chunks))

def feature_matrix_hdf5(smis: Iterable[str], size: int, *,
                        featurizer: Featurizer = Featurizer(),
                        name: str = 'fps',
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
        the set of indices in xs containing invalid inputs
    """
    fps_h5 = str(Path(path)/f'{name}.h5')

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

                fps_dset[i] = fp
                i += 1
        # original dataset size included potentially invalid xs
        valid_size = size - len(invalid_idxs)
        if valid_size != size:
            fps_dset.resize(valid_size, axis=0)

    return fps_h5, invalid_idxs
