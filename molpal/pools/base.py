import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gzip
from itertools import repeat
import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Type, Union

import h5py
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from molpal.encoder import Encoder
from molpal.pools.cluster import cluster_fps_h5
from molpal.pools import fingerprints

# a Mol is a SMILES string, a fingerprint, and an optional cluster ID
Mol = Tuple[str, np.ndarray, Optional[int]]

class MoleculePool(Sequence[Mol]):
    """A MoleculePool is a sequence of molecules in a virtual chemical library

    By default, a MoleculePool eagerly calculates the uncompressed feature
    representations of its entire library and stores these in an hdf5 file.
    If this is undesired, consider using a LazyMoleculePool, which calculates
    these representations only when needed (and recomputes them as necessary.)

    A MoleculePool is most accurately described as a Sequence of Mols, and the
    custom class is necessary for both utility and memory purposes. Its main
    purpose is to hide the storage of large numbers of SMILES strings and
    molecular fingerprints on disk.

    Attributes
    ----------
    library : str
        the filepath of a (compressed) CSV containing the virtual library
    size : int
        the number of molecules in the pool
    invalid_lines : Set[int]
        the set of invalid lines in the libarary file. Will be None until the 
        pool is validated
    title_line : bool
        whether there is a title line in the library file
    delimiter : str
        the column delimiter in the library file
    smiles_col : int
        the column containing the SMILES strings in the library file
    fps : str
        the filepath of an hdf5 file containing the precomputed fingerprints
    smis_ : Optional[List[str]]
        a list of SMILES strings in the pool. None if no caching
    cluster_ids_ : Optional[List[int]]
        the cluster ID for each molecule in the molecule. None if not clustered
    cluster_sizes : Dict[int, int]
        the size of each cluster in the pool. None if not clustered
    chunk_size : int
        the size of each chunk in the hdf5 file
    open_ : Callable[..., TextIO]
        an alias for open or gzip.open, depending on the format of the library 
    verbose : int
        the level of output the pool should print while initializing

    Parameters
    ----------
    library : str
    title_line : bool, default=True
    delimiter : str, default=','
    smiles_col : int, default=0
    fps : Optional[str] (Default = None)
        the filepath of an hdf5 file containing the precomputed fingerprints.
        If specified, a user assumes the following:
        1.  the ordering of the fingerprints matches the ordering in the
            library file
        2.  the encoder used to generate the fingerprints is the same
            as the one passed to the model
        If None, the MoleculePool will generate this file automatically 
    encoder : Encoder (Default = Encoder())
        the encoder to use when calculating fingerprints
    ncpu : int (Default = 1)
        the number of cores to parallelize fingerprint calculation over.
        A value of -1 uses all available cores.
    cache : bool (Default = False)
        whether to cache the SMILES strings in memory
    validated : bool (Default = False)
        whether the pool has been validated already. If True, the user
        accepts the risk of an invalid molecule raising an exception later
        on. Mostly useful for multiple runs on a pool that has been
        manually validated and pruned.
    cluster : bool (Default = False)
        whether to cluster the library
    ncluster : int (Default = 100)
        the number of clusters to form. Only used if cluster is True
    nn_threshold : Optional[float], default=None
        the distance threshold below which to consider pool members neighbors.
        Setting this will calculate all pool neighbors.
    path : str
        the path under which the h5 file should be written
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, library: str, title_line: bool = True,
                 delimiter: str = ',', smiles_col: int = 0,
                 fps: Optional[str] = None,
                 encoder: Encoder = Encoder(), ncpu: int = 1,
                 cache: bool = False, validated: bool = False,
                 cluster: bool = False, ncluster: int = 100,
                 nn_threshold: Optional[float] = None,
                 path: str = '.', verbose: int = 0, **kwargs):
        self.library = library
        self.title_line = title_line
        self.delimiter = delimiter
        self.smiles_col = smiles_col
        self.verbose = verbose

        if Path(library).suffix == '.gz':
            self.open_ = partial(gzip.open, mode='rt')
        else:
            self.open_ = open

        self.fps_ = fps
        self.invalid_lines = None
        self.ncpu = ncpu

        self.smis_ = None
        self.cluster_ids_ = None
        self.cluster_sizes = None
        
        self.chunk_size = self._encode_mols(encoder, ncpu, path)
        self.size = self._validate_and_cache_smis(cache, validated)

        if cluster:
            self._cluster_mols(ncluster)

        self.nn_threshold = nn_threshold

        self._mol_generator = None

    @property
    def neighbors(self) -> List[np.ndarray]:
        """the indices of each pool member's neighbors

        Raises
        ------
        AttributeError
            if nn_threshold is not set
        """
        try:
            return self.__neighbors
        except AttributeError:
            raise AttributeError(
                'nn_threshold must be set before accessing self.neighbors'
            )
    
    @property
    def nn_threshold(self) -> Optional[float]:
        """the distance threshold below which pool members are considered 
        neighbors. None if neighbors have not been calculated"""
        return self.__nn_threshold

    @nn_threshold.setter
    def nn_threshold(self, threshold: Optional[float]):
        """set the distance threshold below which to consider pool members neighbors and calculate the neighbors accordingly."""
        if threshold is not None:
            self.__neighbors = fingerprints.neighbors(self.smis(), threshold)
        self.__nn_threshold = threshold

    def __len__(self) -> int:
        """The number of valid pool inputs"""
        return self.size

    def __iter__(self) -> Iterator[Mol]:
        """Return an iterator over the molecule pool.

        Not recommended for use in open constructs.
        I.e., don't explicitly call this method unless you plan to exhaust the 
        full iterator.
        """
        self._mol_generator = zip(
            self.smis(),
            self.fps(),
            self.cluster_ids() or repeat(None)
        )
        return self

    def __next__(self) -> Mol:
        """Iterate to the next molecule in the pool. Only usable after
        calling iter()"""
        if self._mol_generator:
            return next(self._mol_generator)

        raise StopIteration

    def __contains__(self, smi: str) -> bool:
        for smi_ in self.smis():
            if smi_ == smi:
                return True

        return False

    def __getitem__(self, idx) -> Union[List[Mol], Mol]:
        """Get a molecule with fancy indexing."""
        if isinstance(idx, tuple):
            mols = []
            for i in idx:
                item = self.__getitem__(i)
                if isinstance(item, List):
                    mols.extend(item)
                else:
                    mols.append(item)
            return mols

        if isinstance(idx, slice):
            idx.start = idx.start if idx.start else 0
            if idx.start < -len(self):
                idx.start = 0
            elif idx.start < 0:
                idx.start += len(self)

            idx.stop = idx.stop if idx.stop else len(self)
            if idx.stop > len(self):
                idx.stop = len(self)
            if idx.stop < 0:
                idx.stop = max(0, idx.stop + len(self))

            idx.step = idx.step if idx.step else 1

            idxs = list(range(idx.start, idx.stop, idx.step))
            idxs = sorted([i + len(self) if i < 0 else i for i in idxs])
            return self.get_mols(idxs)

        if isinstance(idx, (int, str)):
            return (
                self.get_smi(idx),
                self.get_fp(idx),
                self.get_cluster_id(idx)
            )

        raise TypeError('pool indices must be integers, slices, '
                        + f'or tuples thereof. Received: {type(idx)}')

    def get_smi(self, idx: int) -> str:
        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        if self.smis_:
            return self.smis_[idx]

        while idx in self.invalid_lines:
            # external indices correspond internally to the line immediately
            # following the invalid line(s)
            idx += 1

        # should test out using islice to advance instead of manually
        for i, smi in enumerate(self.smis()):
            if i==idx:
                return smi

        assert False    # shouldn't reach this point

    def get_fp(self, idx: int) -> np.ndarray:
        idx = idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        with h5py.File(self.fps_, mode='r') as h5fid:
            fps = h5fid['fps']
            return fps[idx]

        assert False    # shouldn't reach this point

    def get_cluster_id(self, idx: int) -> Optional[int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        if self.cluster_ids_:
            return self.cluster_ids_[idx]

        return None

    def get_mols(self, idxs: Sequence[int]) -> List[Mol]:
        return list(zip(
            self.get_smis(idxs),
            self.get_fps(idxs),
            self.get_cluster_ids(idxs) or repeat(None)
        ))

    def get_smis(self, idxs: Sequence[int]) -> List[str]:
        """Get the SMILES strings for the given indices

        NOTE: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        if self.smis:
            idxs = sorted(idxs)
            smis = [self.smis_[i] for i in sorted(idxs)]
        else:
            idxs = set(idxs)
            smis = [smi for i, smi in enumerate(self.smis()) if i in idxs]

        return smis

    def get_fps(self, idxs: Sequence[int]) -> np.ndarray:
        """Get the uncompressed feature representations for the given indices

        NOTE: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        idxs = sorted(idxs)
        with h5py.File(self.fps_, 'r') as h5fid:
            fps = h5fid['fps']
            enc_mols = fps[idxs]

        return enc_mols

    def get_cluster_ids(self, idxs: Sequence[int]) -> Optional[List[int]]:
        """Get the cluster_ids for the given indices, if the pool is
        clustered. Otherwise, return None

        NOTE: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        if self.cluster_ids:
            idxs = sorted(idxs)
            return [self.cluster_ids_[i] for i in idxs]

        return None

    def smis(self) -> Iterator[str]:
        """A generator over pool molecules' SMILES strings
        
        Yields
        ------
        smi : str
            a molecule's SMILES string
        """
        if self.smis_:
            for smi in self.smis_:
                yield smi
        else:
            with self.open_(self.library) as fid:
                reader = csv.reader(fid, delimiter=self.delimiter)
                if self.title_line:
                    next(reader)

                if self.invalid_lines:
                    for i, row in enumerate(reader):
                        if i in self.invalid_lines:
                            continue
                        yield row[self.smiles_col]
                else:
                    for row in reader:
                        yield row[self.smiles_col]

    def fps(self) -> Iterator[np.ndarray]:
        """Return a generator over pool molecules' feature representations
        
        Yields
        ------
        fp : np.ndarray
            a molecule's fingerprint
        """
        with h5py.File(self.fps_, 'r') as h5fid:
            fps = h5fid['fps']
            for fp in fps:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        """Return a generator over batches of pool molecules' feature
        representations
        
        If operating on batches of fingerpints, it is likely more preferable
        to use this method in order to ensure efficient chunk access from the
        internal HDF5 file

        Yields
        ------
        fps_batch : np.ndarray
            a batch of molecular fingerprints
        """
        with h5py.File(self.fps_, 'r') as h5fid:
            fps = h5fid['fps']
            for i in range(0, len(fps), self.chunk_size):
                yield fps[i:i+self.chunk_size]

    def cluster_ids(self) -> Optional[Iterator[int]]:
        """If the pool is clustered, return a generator over pool inputs'
        cluster IDs. Otherwise, return None"""
        if self.cluster_ids_:
            def return_gen(cids):
                for cid in cids:
                    yield cid
            return return_gen(self.cluster_ids_)

        return None

    def _encode_mols(self, encoder: Type[Encoder], 
                     ncpu: int, path: str) -> int:
        """Precalculate the fingerprints of the library members, if necessary.

        Parameters
        ----------
        encoder : Type[Encoder]
            the encoder to use for generating the fingerprints
        ncpu : int
            the number of cores to parallelize fingerprint calculation over
        path : str
            the path to which the fingerprints file should be written
        
        Returns
        -------
        chunk_size : int
            the length of each chunk in the HDF5 file
        
        Side effects
        ------------
        (sets) self.fps_ : str
            the filepath of the h5 file containing the fingerprints
        (sets) self.invalid_lines : Set[int]
            the set of invalid lines in the library file
        (sets) self.size : int
            the number of valid SMILES strings in the library
        """
        if self.fps_ is None:
            if self.verbose > 0:
                print('Precalculating feature matrix ...', end=' ')

            total_size = sum(1 for _ in self.smis())
            self.fps_, self.invalid_lines = fingerprints.feature_matrix_hdf5(
                self.smis(), total_size, ncpu=self.ncpu,
                encoder=encoder, name=Path(self.library).stem, path=path
            )
            if self.verbose > 0:
                print('Done!')
                print(f'Feature matrix was saved to "{self.fps_}"', flush=True)
                print(f'Detected {len(self.invalid_lines)} invalid SMILES!')
        else:
            if self.verbose > 0:
                print(f'Using feature matrix from "{self.fps_}"', flush=True)

        with h5py.File(self.fps_, 'r') as h5f:
            fps = h5f['fps']
            chunk_size = fps.chunks[0]
            self.size = len(fps)

        return chunk_size

    def _validate_and_cache_smis(self, cache: bool = False,
                                 validated: bool = False) -> int:
        """Validate all the SMILES strings in the pool and return the length
        of the validated pool

        Parameters
        ----------
        cache : bool (Default = False)
            whether to cache the SMILES strings as well
        validated : bool (Default = False)
            whether the pool has been validated already. If True, the user 
            accepts the risk of an invalid molecule raising an exception later
            on. Mostly useful for multiple runs on a pool that has been
            previously validated and pruned.

        Returns
        -------
        size : int
            the size of the validated pool
        
        Side effects
        ------------
        (sets) self.smis_ : List[str]
            if cache is True
        (sets) self.invalid_lines
            if cache is False and was not previously set by _encode_mols()
        """
        if self.invalid_lines is not None and not cache:
            # the pool has already been validated if invalid_lines is set
            return len(self)

        if self.verbose > 0:
            print('Validating SMILES strings ...', end=' ')

        self.invalid_lines = set()
        smis = self.smis()

        if validated:
            if cache:
                self.smis_ = [smi for smi in tqdm(smis, desc='Caching')]
        else:
            with ProcessPoolExecutor(max_workers=self.ncpu) as pool:
                valid_smis = pool.map(validate_smi, smis, chunksize=256)
                if cache:
                    self.smis_ = []
                    for i, smi in tqdm(enumerate(valid_smis), unit='smi',
                                            desc='Validating', smoothing=0.):
                        if smi is None:
                            self.invalid_lines.add(i)
                        else:
                            self.smis_.append(smi)
                else:
                    for i, smi in tqdm(enumerate(valid_smis), unit='smi',
                                            desc='Validating', smoothing=0.):
                        if smi is None:
                            self.invalid_lines.add(i)

        if self.verbose > 0:
            print('Done!', flush=True)
        if self.verbose > 1:
            print(f'Detected {len(self.invalid_lines)} invalid SMILES')

        return sum(1 for _ in self.smis()) - len(self.invalid_lines)

    def _cluster_mols(self, ncluster: int) -> None:
        """Cluster the molecules in the library.

        Parameters
        ----------
        ncluster : int
            the number of clusters to form
        
        Side effects
        ------------
        (sets) self.cluster_ids : List[int]
            a list of cluster IDs that is parallel to the valid SMILES strings
        (sets) self.cluster_sizes : Counter[int, int]
            a mapping from cluster ID to the number of molecules in that cluster
        """
        self.cluster_ids_ = cluster_fps_h5(self.fps_, ncluster=ncluster)
        self.cluster_sizes = Counter(self.cluster_ids_)

def validate_smi(smi):
    return smi if Chem.MolFromSmiles(smi) else None

class EagerMoleculePool(MoleculePool):
    """Alias for a MoleculePool"""
