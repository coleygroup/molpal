from typing import Iterator, Optional, Sequence, Union

import numpy as np
import ray

from molpal.utils import batches
from molpal.featurizer import Featurizer, feature_matrix
from molpal.pools.base import MoleculePool, PruneMethod


class LazyMoleculePool(MoleculePool):
    """A LazyMoleculePool does not precompute fingerprints for the pool

    Attributes (only differences with EagerMoleculePool are shown)
    ----------
    featurizer : Featurizer
        an Featurizer to generate uncompressed representations on the fly
    fps : None
        no fingerprint file is stored for a LazyMoleculePool
    chunk_size : int
        the buffer size of calculated fingerprints during pool iteration
    cluster_ids : None
        no clustering can be performed for a LazyMoleculePool
    cluster_sizes : None
        no clustering can be performed for a LazyMoleculePool
    """

    def get_fp(self, idx: int) -> np.ndarray:
        smi = self.get_smi(idx)
        return self.featurizer(smi)

    def get_fps(self, idxs: Sequence[int]) -> np.ndarray:
        smis = self.get_smis(idxs)
        return np.array([self.featurizer(smi) for smi in smis])

    def fps(self) -> Iterator[np.ndarray]:
        for fps_batch in self.fps_batches():
            for fp in fps_batch:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        for smis in batches(self.smis(), self.chunk_size):
            yield np.array(feature_matrix(smis, self.featurizer, True))

    def prune(
        self,
        k: Union[int, float],
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        prune_method: PruneMethod = PruneMethod.GREEDY,
        l: Optional[Union[int, float]] = None,
        beta: float = 2.,
        max_fp: Optional[Union[int, float]] = None,
        min_hit_prob: float = 0.025
    ) -> float:
        if isinstance(k, float):
            k = int(k * len(Y_mean))
        if k < 1:
            raise ValueError(f"hit threshold (k) must be positive! got: {k}")

        if prune_method == PruneMethod.GREEDY:
            idxs = self.prune_greedy(Y_mean, l)
        elif prune_method == PruneMethod.UCB:
            idxs = self.prune_ucb(Y_mean, Y_var, l, beta)
        elif prune_method == PruneMethod.EFP:
            idxs = self.prune_max_fp(k, Y_mean, Y_var, max_fp)
        elif prune_method == PruneMethod.PROB:
            idxs = self.prune_prob(Y_mean, Y_var, l, min_hit_prob)
        
        self.smis_ = self.get_smis(idxs)
        self.size = len(self.smis_)

        return MoleculePool.expected_positives_pruned(k, Y_mean, Y_var, idxs)

    def _encode_mols(self, *args, **kwargs):
        """Do not precompute any feature representations"""
        self.chunk_size = int(ray.cluster_resources()["CPU"] * 4096)
        self.fps_ = None

    def _cluster_mols(self, *args, **kwargs) -> None:
        """A LazyMoleculePool can't cluster molecules

        Doing so would require precalculating all uncompressed representations,
        which is what a LazyMoleculePool is designed to avoid. If clustering
        is desired, use the base (Eager)MoleculePool
        """
        print(
            "WARNING: Clustering is not possible for a LazyMoleculePool.",
            "No clustering will be performed.",
        )
