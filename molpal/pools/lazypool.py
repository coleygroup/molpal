from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Iterator, Sequence, Type

import numpy as np

from molpal.encoders import Encoder
from molpal.pools.base import MoleculePool, Mol

# encoder = AtomPairFingerprinter()
# def smi_to_fp(smi):
#     return encoder.encode_and_uncompress(smi)

class LazyMoleculePool(MoleculePool):
    """A LazyMoleculePool does not precompute fingerprints for the pool

    Attributes (only differences with EagerMoleculePool are shown)
    ----------
    encoder : Type[Encoder]
        an encoder to generate uncompressed representations on the fly
    fps : None
        no fingerprint file is stored for a LazyMoleculePool
    chunk_size : int
        the buffer size of calculated fingerprints during pool iteration
    cluster_ids : None
        no clustering can be performed for a LazyMoleculePool
    cluster_sizes : None
        no clustering can be performed for a LazyMoleculePool
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.chunk_size = 100 * self.njobs

    def __iter__(self) -> Iterator[Mol]:
        """Return an iterator over the molecule pool.

        Not recommended for use in open constructs. I.e., don't explicitly
        call this method unless you plan to exhaust the full iterator.
        """
        self._mol_generator = zip(self.smis(), self.fps())
        
        return self

    def __next__(self) -> Mol:
        if self._mol_generator is None:
            raise StopIteration

        return next(self._mol_generator)

    def get_fp(self, idx: int) -> np.ndarray:
        smi = self.get_smi(idx)
        return self.encoder.encode_and_uncompress(smi)

    def get_fps(self, idxs: Sequence[int]) -> np.ndarray:
        smis = self.get_smis(idxs)
        return np.array([self.encoder.encode_and_uncompress(smi)
                         for smi in smis])

    def fps(self) -> Iterator[np.ndarray]:
        for fps_chunk in self.fps_batches():
            for fp in fps_chunk:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        # buffer of chunk of fps into memory for faster iteration
        job_chunk_size = self.chunk_size // self.njobs
        smis = iter(self.smis())
        smis_chunks = iter(lambda: list(islice(smis, self.chunk_size)), [])

        with ProcessPoolExecutor(max_workers=self.njobs) as pool:
            for smis_chunk in smis_chunks:
                fps_chunk = pool.map(self.encoder.encode_and_uncompress, 
                                     smis_chunk, chunksize=job_chunk_size)
                yield fps_chunk

    def _encode_mols(self, encoder: Type[Encoder], njobs: int, 
                     **kwargs) -> None:
        """
        Side effects
        ------------
        (sets) self.encoder : Type[Encoder]
            the encoder used to generate molecular fingerprints
        (sets) self.njobs : int
            the number of jobs to parallelize fingerprint buffering over
        """
        self.encoder = encoder
        self.njobs = njobs

    def _cluster_mols(self, *args, **kwargs) -> None:
        """A LazyMoleculePool can't cluster molecules

        Doing so would require precalculating all uncompressed representations,
        which is what a LazyMoleculePool is designed to avoid. If clustering
        is desired, use the base (Eager)MoleculePool
        """
        print('WARNING: Clustering is not possible for a LazyMoleculePool.',
              'No clustering will be performed.')
