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

from .cluster import cluster_fps_h5
from . import fingerprints
from ..encoders import Encoder, AtomPairFingerprinter

# a Mol is a SMILES string, a fingerprint, and an optional cluster ID
Mol = Tuple[str, np.ndarray, Optional[int]]

try:
    MAX_CPU = len(os.sched_getaffinity(0))
except AttributeError:
    MAX_CPU = os.cpu_count()

class MoleculePool(Sequence[Mol]):
    """A MoleculePool is a sequence of molecules in a virtual chemical library

    By default, a MoleculePool eagerly calculates the uncompressed feature
    representations of its entire library and stores these in an hdf5 file.
    If this is undesired, consider using a LazyMoleculePool, which calculates
    these representations only when needed (and recomputes them as necessary.)

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
        the filepath of an hdf5 file containing precomputed fingerprints
    smis : Optional[List[str]]
        a list of SMILES strings in the pool. None if no caching
    cluster_ids : Optional[List[int]]
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
    fps : Optional[str]
        the filepath of an hdf5 file containing precalculated fingerprints
        of the library.
        A user assumes the following:
        1. the ordering of the fingerprints matches the ordering in the
            library file
        2. the encoder used to generate the fingerprints is the same
            as the one passed to the model
    enc : Type[Encoder] (Default = AtomPairFingerprinter)
        the encoder to use when calculating fingerprints
    njobs : int (Default = -1)
        the number of jobs to parallelize fingerprint calculation over.
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
        the number of clusters to form
    path : str
        the path to which the h5 file should be written
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, library: str, title_line: bool = True,
                 delimiter: str = ',', smiles_col: int = 0,
                 fps: Optional[str] = None,
                 enc: Type[Encoder] = AtomPairFingerprinter(), njobs: int = 4,
                 cache: bool = False, validated: bool = False,
                 cluster: bool = False, ncluster: int = 100,
                 path: str = '.', verbose: int = 0, **kwargs):
        self.library = library
        self.title_line = title_line
        self.delimiter = delimiter
        self.verbose = verbose

        if Path(library).suffix == '.gz':
            self.open_ = partial(gzip.open, mode='rt')
        else:
            self.open_ = open
            
        self.smiles_col = smiles_col

        self.fps = fps
        self.invalid_lines = None
        self.njobs = njobs
        self.chunk_size = self._encode_mols(enc, njobs, path)

        self.smis = None
        self.d_smi_idx = {}
        self.size = self._validate_and_cache_smis(cache, validated)

        self.cluster_ids = None
        self.cluster_sizes = None
        if cluster:
            self._cluster_mols(ncluster)

        self._mol_generator = None

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Mol]:
        """Return an iterator over the molecule pool.

        Not recommended for use in open constructs. I.e., don't call
        explicitly call this method unless you plan to exhaust the full
        iterator.
        """
        self._mol_generator = zip(
            self.gen_smis(),
            self.gen_enc_mols(),
            self.gen_cluster_ids() or repeat(None)
        )
        return self

    def __next__(self) -> Mol:
        """Iterate to the next molecule in the pool. Only usable after
        calling iter()"""
        if self._mol_generator:
            return next(self._mol_generator)

        raise StopIteration

    def __contains__(self, smi: str) -> bool:
        return hash(smi) in self.d_smi_idx

        # if self.smis is not None:
        #     return smi in self.smis


        # with self.open_(self.library) as fid:
        #     reader = csv.reader(fid)
        #     if self.title_line:
        #         next(reader)

        #     for row in reader:
        #         if smi == row[self.smiles_col]:
        #             return True

        # return False

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
                self.get_enc_mol(idx),
                self.get_cluster_id(idx)
            )

        raise TypeError('pool indices must be integers, slices, '
                        + f'or tuples thereof. Received: {type(idx)}')

    def get_smi(self, smi_or_idx: Union[int, str]) -> str:
        if isinstance(smi_or_idx, str):
            return smi_or_idx

        idx = smi_or_idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        if self.smis:
            return self.smis[idx]

        # skip invalid lines
        while idx in self.invalid_lines:
            idx += 1

        with self.open_(self.library) as fid:
            reader = csv.reader(fid, delimiter=self.delimiter)
            if self.title_line:
                next(reader)

            for i, row in enumerate(reader):
                if i == idx:
                    return row[self.smiles_col]

        assert False    # shouldn't reach this point

    def get_enc_mol(self, smi_or_idx: Union[str, int]) -> np.ndarray:
        if isinstance(smi_or_idx, str):
            idx = self.d_smi_idx[hash(smi_or_idx)]
        else:
            idx = smi_or_idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        with h5py.File(self.fps) as h5f:
            fps = h5f['fps']
            return fps[idx]

        assert False    # shouldn't reach this point

    def get_cluster_id(self, smi_or_idx: Union[str, int]) -> Optional[int]:
        if isinstance(smi_or_idx, str):
            idx = self.d_smi_idx[hash(smi_or_idx)]
        else:
            idx = smi_or_idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f'pool index(={idx}) out of range')

        if self.cluster_ids:
            return self.cluster_ids[idx]

        return None

    def get_mols(self, idxs: Sequence[int]) -> Iterator[Mol]:
        return zip(
            self.get_smis(idxs),
            self.get_enc_mols(idxs),
            self.get_cluster_ids(idxs) or repeat(None)
        )

    def get_smis(self, idxs: Sequence[int]) -> List[str]:
        """Get the SMILES strings for the given indices

        WARNING: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        if self.smis:
            idxs = sorted(idxs)
            smis = [self.smis[i] for i in sorted(idxs)]
        else:
            idxs = set(idxs)
            smis = [smi for i, smi in enumerate(self.gen_smis()) if i in idxs]

        return smis

    def get_enc_mols(self, idxs: Sequence[int]) -> np.ndarray:
        """Get the uncompressed feature representations for the given indices

        WARNING: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        idxs = sorted(idxs)
        with h5py.File(self.fps, 'r') as h5f:
            fps = h5f['fps']
            enc_mols = fps[idxs]

        return enc_mols

    def get_cluster_ids(self, idxs: Sequence[int]) -> Optional[List[int]]:
        """Get the cluster_ids for the given indices, if the pool is
        clustered. Otherwise, return None

        WARNING: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f'Pool index out of range: {idxs}')

        if self.cluster_ids:
            idxs = sorted(idxs)
            return [self.cluster_ids[i] for i in idxs]

        return None

    def gen_smis(self) -> Iterator[str]:
        """Return a generator over pool molecules' SMILES strings"""
        if self.smis:
            for smi in self.smis:
                yield smi
        else:
            with self.open_(self.library) as fid:
                reader = csv.reader(fid, delimiter=self.delimiter)
                if self.title_line:
                    next(reader)

                for i, row in enumerate(reader):
                    if i in self.invalid_lines:
                        continue
                    yield row[self.smiles_col]

    def gen_enc_mols(self) -> Iterator[np.ndarray]:
        """Return a generator over pool molecules' feature representations"""
        with h5py.File(self.fps, 'r') as h5f:
            fps = h5f['fps']
            for fp in fps:
                yield fp

    def gen_batch_enc_mols(self) -> Iterator[np.ndarray]:
        """Return a generator over batches of pool molecules' feature
        representations"""
        with h5py.File(self.fps, 'r') as h5f:
            fps = h5f['fps']
            for i in range(0, len(fps), self.chunk_size):
                yield fps[i:i+self.chunk_size]

    def gen_cluster_ids(self) -> Optional[Iterator[int]]:
        """If the pool is clustered, return a generator over pool inputs'
        cluster IDs. Otherwise, return None"""
        if self.cluster_ids:
            def return_gen(cids):
                for cid in cids:
                    yield cid
            return return_gen(self.cluster_ids)

        return None

    def _encode_mols(self, enc: Type[Encoder], njobs: int, path: str) -> int:
        """Precalculate the fingerprints of the library members, if necessary.

        Parameters
        ----------
        delimiter : str
            the column separator in the library file
        enc : Type[Encoder]
            the encoder to use for generating the fingerprints
        njobs : int
            the number of jobs to parallelize fingerprint calculation over
        path : str
            the path two which the fingerprints file should be written
        
        Returns
        -------
        chunk_size : int
            the size of each chunk in the h5 file containing the fingerprints
        
        Side effects
        ------------
        (sets) self.fps : str
            the filepath of the h5 file containing the fingerprints
        (sets) self.invalid_lines : Set[int]
            the set of invalid lines in the library file
        (sets) self.size : int
            the number of valid SMILES strings in the library
        """
        if self.fps is None:
            if self.verbose > 0:
                print('Precalculating fingerprints ...', end=' ')

            self.fps, self.invalid_lines = fingerprints.parse_smiles_par(
                self.library, delimiter=self.delimiter,
                smiles_col=self.smiles_col, title_line=self.title_line, 
                encoder_=enc, njobs=njobs, path=path
            )

            if self.verbose > 0:
                print('Done!')
                print(f'Molecular fingerprints were saved to "{self.fps}"')
        else:
            if self.verbose > 0:
                print(f'Using presupplied fingerprints from "{self.fps}"')

        with h5py.File(self.fps, 'r') as h5f:
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
            manually validated and pruned.

        Returns
        -------
        size : int
            the size of the validated pool
        
        Side effects
        ------------
        (sets) self.smis : List[str]
            if cache is True
        (sets) self.invalid_lines
            if cache is False and was not previously set by _encode_mols()
        """
        if self.invalid_lines is not None and not cache:
            # the pool has already been validated if invalid_lines is set
            return len(self)

        if self.verbose > 0:
            print('Validating SMILES strings ...', end=' ', flush=True)

        self.invalid_lines = set()
        with self.open_(self.library) as fid:
            reader = csv.reader(fid, delimiter=self.delimiter)
            if self.title_line:
                next(reader)

            smis = (row[self.smiles_col] for row in reader)
            if validated:
                if cache:
                    self.smis = [smi for smi in tqdm(smis, desc='Caching')]
                    self.d_smi_idx = {hash(smi): i 
                                      for i, smi in enumerate(self.smis)}
                else:
                    self.d_smi_idx = {hash(smi): i 
                                      for i, smi in enumerate(smis)}
            else:
                with ProcessPoolExecutor(max_workers=self.njobs) as pool:
                    smis_mols = pool.map(smi_to_mol, smis, chunksize=256)
                    if cache:
                        self.smis = []
                        for i, smi_mol in tqdm(enumerate(smis_mols), unit='smi',
                                               desc='Validating', smoothing=0.):
                            smi, mol = smi_mol
                            if mol is None:
                                self.invalid_lines.add(i)
                            else:
                                self.smis.append(smi)
                        self.d_smi_idx = {hash(smi): i 
                                          for i, smi in enumerate(self.smis)}
                    else:
                        for i, smi_mol in tqdm(enumerate(smis_mols), unit='smi',
                                               desc='Validating', smoothing=0.):
                            smi, mol = smi_mol
                            if mol is None:
                                self.invalid_lines.add(i)
                            else:
                                self.d_smi_idx[hash(smi)] = i
                                size += 1

        if self.verbose > 0:
            print('Done!')
        if self.verbose > 1:
            print(f'Detected {len(self.invalid_lines)} invalid SMILES strings')

        return len(self.d_smi_idx)

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
        self.cluster_ids = cluster_fps_h5(self.fps, ncluster=ncluster)
        self.cluster_sizes = Counter(self.cluster_ids)

def smi_to_mol(smi):
    return smi, Chem.MolFromSmiles(smi)

class EagerMoleculePool(MoleculePool):
    """Alias for a MoleculePool"""
