import csv
from collections import Counter
from enum import auto
from functools import partial
import gzip
from itertools import islice, repeat
import re
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import uuid

import h5py
import numpy as np
import ray
from rdkit import Chem, RDLogger
from scipy.stats import norm
from tqdm import tqdm

from molpal.featurizer import Featurizer
from molpal.pools import cluster, fingerprints
from molpal.utils import AutoName, batches

RDLogger.DisableLog("rdApp.*")

Mol = Tuple[str, np.ndarray, Optional[int]]

CXSMILES_PATTERN = re.compile(r"\s\|.*\|")

class PruneMethod(AutoName):
    EFP = auto()
    PROB = auto()
    TOP = auto()

class MoleculePool(Sequence):
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
    libraries : str
        the filepaths of (compressed) CSVs containing the pool members. NOTE: each file must be
        in the same format. That is, if the first file is a TSV of CXSMILES strings with no
        title line, then the second file must also be a TSV of CXSMILES strings with no
        title line, and so on for each successive file.
    size : int
        the total number of molecules in the pool
    invalid_idxs : Set[int]
        the set of invalid indices in the libarary files. Will be None until
        the pool is validated
    title_line : bool
        whether there is a title line in each library file
    delimiter : str
        the column delimiters in the library files
    smiles_col : int
        the column containing the SMILES strings in the library files
    fps : str
        the filepath of an HDF5 file containing the precomputed fingerprint
        for each pool member
    smis_ : Optional[List[str]]
        a list of SMILES strings in the pool. None if no caching
    cluster_ids_ : Optional[List[int]]
        the cluster ID for each molecule in the molecule. None if not clustered
    cluster_sizes : Dict[int, int]
        the size of each cluster in the pool. None if not clustered
    chunk_size : int
        the size of each chunk in the HDF5 file
    verbose : int
        the level of output the pool should print while initializing

    Parameters
    ----------
    library : Iterable[str]
    title_line : bool, default=True
    delimiter : str, default=','
    smiles_col : int, default=0
    fps : Optional[str], default=None
        the filepath of an hdf5 file containing the precomputed fingerprints.
        If specified, a user assumes the following:
        1. the ordering of the fingerprints matches the ordering in the
            library file
        2. the featurizer used to generate the fingerprints is the same
            as the one passed to the model
        If None, the MoleculePool will generate this file automatically
    featurizer : Featurizer, default=Featurizer()
        the featurizer to use when calculating fingerprints
    cache : bool, default=False
        whether to cache the SMILES strings in memory
    validated : bool, default=False
        whether the pool has been validated already. If True, the user
        accepts the risk of an invalid molecule raising an exception later
        on. Mostly useful for multiple runs on a pool that has been
        manually validated and pruned.
    cluster : bool, default=False
        whether to cluster the library
    ncluster : int, default=100
        the number of clusters to form. Only used if cluster is True
    fps_path : Optional[str], default=None
        the path under which the HDF5 file should be written. By default,
        will write the fingerprints HDF5 file under the same directory as the
        first library file
    verbose : int , default=0
    **kwargs
        additional and unused keyword arguments
    """

    def __init__(
        self,
        libraries: Iterable[str],
        title_line: bool = True,
        delimiter: str = ",",
        smiles_col: int = 0,
        cxsmiles: bool = False,
        fps: Optional[str] = None,
        fps_path: Optional[str] = None,
        featurizer: Featurizer = Featurizer(),
        cache: bool = False,
        invalid_idxs: Optional[Iterable[int]] = None,
        cluster: bool = False,
        ncluster: int = 100,
        verbose: int = 0,
        **kwargs,
    ):
        if not ray.is_initialized():
            print("No ray cluster detected! attempting to connect!")
            try:
                ray.init("auto")
            except ConnectionError:
                ray.init()

        self.libraries = list(libraries)
        self.title_line = title_line
        self.delimiter = delimiter
        self.smiles_col = smiles_col
        self.cxsmiles = cxsmiles
        self.verbose = verbose

        self.smis_ = None
        self.fps_ = fps
        self.fps_path = fps_path
        self.featurizer = featurizer

        self.invalid_idxs = set(invalid_idxs) if invalid_idxs is not None else None
        self.size = None
        self.chunk_size = None
        self._encode_mols(featurizer, fps_path)
        self._validate_and_cache_smis(cache)

        self.cluster_ids_ = None
        self.cluster_sizes = None
        if cluster:
            self._cluster_mols(ncluster)

    def __len__(self) -> int:
        """The number of valid pool inputs"""
        return self.size

    def __iter__(self) -> Iterator[Mol]:
        """Return an iterator over the molecule pool.

        NOTE: Not recommended for use in open constructs. I.e., don't
        explicitly call this method unless you plan to exhaust the full iterator.
        """
        return zip(self.smis(), self.fps(), self.cluster_ids() or repeat(None))

    def __contains__(self, smi: str) -> bool:
        if self.smis_ is not None:
            return smi in set(self.smis_)

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
            return (self.get_smi(idx), self.get_fp(idx), self.get_cluster_id(idx))

        raise TypeError(
            f"pool indices must be integers, slices, or tuples thereof. Received: {type(idx)}"
        )

    def prune(
        self,
        k: Union[int, float],
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        prune_method: PruneMethod = PruneMethod.TOP,
        l: Optional[Union[int, float]] = None,
        beta: float = 2.,
        max_fp: Optional[Union[int, float]] = None,
        min_hit_prob: float = 0.025
    ) -> float:
        """prune the library to the top-k predicted compounds based on their predicted means

        NOTE: if both max_fp and l are specified, pruning will be performed using the top-l
        predicted compounds and NOT by maximizing the number of false positives

        Parameters
        ----------
        k : Union[int, float]
            the percentile or rank above which molecules are classified as "hits" or "positives" 
        Y_mean : np.ndarray
            the predicted mean for each molecule
        Y_var : np.ndarray
            the predicted variance for each molecule
        prune_method : PruneMethod, default=PruneMethod.TOP
            the method by which to prune the pool:
            * PruneMethod.TOP: retain the top-k compounds by predicted mean + beta * predicted
                uncertainty.
            * PruneMethod.EFN: retain a maximal number of expected false positives up to some input
                threshold
            * PruneMethod.PROB: retain all molecules that have probability p > p* of being a "hit"
        l : Union[int, float], default=None
            the number or fraction of the pool to retain after TOP pruning.
        beta : float, default=2.
            the amount by which to multiply the predicted uncertainty before adding to the
            predicted mean
        max_fp : Union[int, float], default=0.01
            the maximal expected number of true negatives to retain for EFN pruning, expressed as
            either a number or fraction of the pool size.
        min_hit_prob : float, default=0.025
            the minimum probability that a compound is a hit for it to be retained in PROB pruning

        Returns
        -------
        float
            the expected number of true positives pruned, where a "positive" constitutes a molecule 
            with a predicted mean inside the top-k
        """
        if isinstance(k, float):
            k = int(k * len(Y_mean))
        if k < 1:
            raise ValueError(f"hit threshold (k) must be positive! got: {k}")

        if prune_method == PruneMethod.TOP:
            idxs = self.prune_top(Y_mean, Y_var, l, beta)
        elif prune_method == PruneMethod.EFP:
            idxs = self.prune_max_fp(k, Y_mean, Y_var, max_fp)
        elif prune_method == PruneMethod.PROB:
            idxs = self.prune_prob(Y_mean, Y_var, l, min_hit_prob)
        
        self.smis_ = self.get_smis(idxs)

        self.fps_, self.invalid_idxs = fingerprints.feature_matrix_hdf5(
            self.smis_,
            l,
            featurizer=self.featurizer,
            name=f"{Path(self.fps_).stem}_pruned_{uuid.uuid4()}",
            path=tempfile.gettempdir()
        )

        with h5py.File(self.fps_, "r") as h5f:
            fps = h5f["fps"]
            self.size = len(fps)
            self.chunk_size = fps.chunks[0]

        return MoleculePool.expected_positives_pruned(k, Y_mean, Y_var, idxs)

    def get_smi(self, idx: int) -> str:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"pool index(={idx}) out of range")

        if self.smis_ is not None:
            return self.smis_[idx]

        return next(islice(self.smis(), idx, idx + 1))

    def get_fp(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"pool index(={idx}) out of range")

        with h5py.File(self.fps_, mode="r") as h5f:
            fps_dset = h5f["fps"]

            return fps_dset[idx]

    def get_cluster_id(self, idx: int) -> Optional[int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"pool index(={idx}) out of range")

        if self.cluster_ids_:
            return self.cluster_ids_[idx]

        return None

    def get_mols(self, idxs: Sequence[int]) -> List[Mol]:
        return list(
            zip(
                self.get_smis(idxs),
                self.get_fps(idxs),
                self.get_cluster_ids(idxs) or repeat(None),
            )
        )

    def get_smis(self, idxs: Sequence[int]) -> List[str]:
        """Get the SMILES strings for the given indices

        NOTE: Returns the list in sorted index order

        Parameters
        ----------
        idxs : Collection[int]
            the indices for which to retrieve the SMILES strings
        """
        if min(idxs) < 0 or max(idxs) >= len(self):
            raise IndexError(f"Pool index out of range: {idxs}")

        if self.smis_ is not None:
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
            raise IndexError(f"Pool index out of range: {idxs}")

        idxs = sorted(idxs)
        with h5py.File(self.fps_, "r") as h5f:
            fps = h5f["fps"]

            return fps[idxs]

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
            raise IndexError(f"Pool index out of range: {idxs}")

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
            for library in self.libraries:
                if self.cxsmiles:
                    for smi in self.read_libary(library):
                        yield CXSMILES_PATTERN.split(smi)[0]
                else:
                    for smi in self.read_libary(library):
                        yield smi

    def read_libary(self, library) -> Iterator[str]:
        """Iterate through the SMILES string in the library using this pool's parsing parameters

        Parameters
        ----------
        library : Union[str, Path]
            the filepath of the library

        Yields
        -------
        Iterator[str]
            the SMILES strings in the library
        """
        if Path(library).suffix == ".gz":
            open_ = partial(gzip.open, mode="rt")
        else:
            open_ = open

        with open_(library) as fid:
            reader = csv.reader(fid, delimiter=self.delimiter)
            if self.title_line:
                next(reader)

            if self.invalid_idxs:
                for i, row in enumerate(reader):
                    if i in self.invalid_idxs:
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
        with h5py.File(self.fps_, "r") as h5f:
            fps = h5f["fps"]
            for fp in fps:
                yield fp

    def fps_batches(self) -> Iterator[np.ndarray]:
        """Return a generator over batches of pool molecules' feature
        representations

        Itt is preferable to use this method when operating on batches of
        fingerprints

        Yields
        ------
        fps_batch : np.ndarray
            a batch of molecular fingerprints
        """
        with h5py.File(self.fps_, "r") as h5f:
            fps = h5f["fps"]
            for i in range(0, len(fps), self.chunk_size):
                yield fps[i : i + self.chunk_size]

    def cluster_ids(self) -> Optional[Iterator[int]]:
        """If the pool is clustered, return a generator over pool inputs'
        cluster IDs. Otherwise, return None"""
        if self.cluster_ids_:

            def return_gen():
                for cid in self.cluster_ids_:
                    yield cid

            return return_gen()

        return None

    def _encode_mols(self, featurizer: Featurizer, path: Optional[str] = None):
        """Precalculate the fingerprints of the library members, if necessary.

        Parameters
        ----------
        featurizer : Featurizer
            the featurizer to use for calculating the fingerprints
        path : Optional[str], default=None
            the path to which the fingerprints file should be written

        Side effects
        ------------
        (sets) self.size : int
            the total number of fingerprints in the fingerprints HDF5 file
        (sets) self.chunk_size : int
            the length of each chunk in the HDF5 file
        (sets) self.fps_ : str
            the filepath of the HDF5 file containing the fingerprints
        (sets) self.invalid_idxs : Set[int]
            the set of invalid lines in the library file
        """
        if self.fps_ is None:
            if self.verbose > 0:
                print("Precalculating feature matrix ...", end=" ")

            total_size = sum(1 for _ in self.smis())

            filename = Path(self.libraries[0])
            while filename.suffix:
                filename = filename.with_suffix("")
            path = path or Path(self.libraries[0]).parent

            self.fps_, self.invalid_idxs = fingerprints.feature_matrix_hdf5(
                self.smis(),
                total_size,
                featurizer=featurizer,
                name=filename.stem,
                path=path,
            )
            if self.verbose > 0:
                print("Done!")
                print(f'Feature matrix was saved to "{self.fps_}"', flush=True)
                print(f"Detected {len(self.invalid_idxs)} invalid SMILES!")
        else:
            if self.verbose > 0:
                print(f'Using feature matrix from "{self.fps_}"', flush=True)

        with h5py.File(self.fps_, "r") as h5f:
            fps = h5f["fps"]
            self.size = len(fps)
            self.chunk_size = fps.chunks[0]

    def _validate_and_cache_smis(self, cache: bool = False) -> int:
        """Validate all the SMILES strings in the pool and return the length
        of the validated pool

        Parameters
        ----------
        cache : bool, default=False
            whether to cache the SMILES strings

        Side effects
        ------------
        (sets) self.smis_ : List[str]
            if cache is True
        (sets) self.size : int
        (sets) self.invalid_idxs : Set[int]
        """
        if self.invalid_idxs is not None:
            if len(self.invalid_idxs) == 0:
                if cache:
                    self.smis_ = [smi for smi in self.smis()]
                    self.size = len(self.smis_)
                else:
                    self.size = self.size or sum(1 for _ in self.smis())
            else:
                if cache:
                    self.smis_ = [
                        smi
                        for i, smi in enumerate(self.smis())
                        if i not in self.invalid_idxs
                    ]
                    self.size = len(self.smis_)
                else:
                    if self.size is None:
                        n_total_smis = sum(1 for _ in self.smis())
                        self.size = n_total_smis - len(self.invalid_idxs)
        else:
            if self.verbose > 0:
                print("Validating SMILES strings ...", end=" ", flush=True)

            valid_smis = validate_smis(self.smis())
            invalid_idxs = set()
            if cache:
                self.smis_ = []
                for i, smi in tqdm(enumerate(valid_smis), desc="Validating"):
                    if smi is None:
                        self.invalid_idxs.add(i)
                    else:
                        self.smis_.append(smi)
            else:
                for i, smi in tqdm(enumerate(valid_smis), desc="Validating"):
                    if smi is None:
                        invalid_idxs.add(i)

            self.size = (i + 1) - len(invalid_idxs)
            self.invalid_idxs = invalid_idxs

            if self.verbose > 0:
                print("Done!", flush=True)
            if self.verbose > 1:
                print(f"Detected {len(self.invalid_idxs)} invalid SMILES")

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
        self.cluster_ids_ = cluster.cluster_fps_h5(self.fps_, ncluster=ncluster)
        self.cluster_sizes = Counter(self.cluster_ids_)

    @staticmethod
    def prune_greedy(Y_mean: np.ndarray, l: Union[int, float]) -> np.ndarray:
        """prune all predictions with mean less than a given threshold
        
        Parameters
        ----------
        Y_mean : np.ndarray
            the predicted means
        l : Union[int, float]
            the percentile or rank of the predicted means from which to calculate a pruning 
            threshold

        Returns
        -------
        np.ndarray
            the indices of the predictions to retain

        Raises
        ------
        ValueError
            if l is a float below 0 or an int less than 1
        """
        return MoleculePool.prune_ucb(Y_mean, np.array([]), l, 0.)

    @staticmethod
    def prune_ucb(
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        l: Union[int, float],
        beta: float = 2.
    ) -> np.ndarray:
        """prune all predictions with mean + beta * sqrt(var) less than a given threshold

        Parameters
        ----------
        Y_mean : np.ndarray
            the predicted means
        Y_var : np.ndarray
            the predicted variances
        l : Union[int, float]
            the percentile or rank of the predicted means from which to calculate a pruning 
            threshold 
        beta : float, default=2
            the number of confidence intervals to add to each predicted mean

        Returns
        -------
        np.ndarray
            the indices of the predictions to retain

        Raises
        ------
        ValueError
            if l is a float below 0 or an int less than 1
        """
        if isinstance(l, float):
            l = int(l * len(Y_mean))
        if l < 1:
            raise ValueError(f"l must be positive! got: {l}")

        Y_ub = Y_mean + beta * np.sqrt(Y_var) if Y_mean.shape == Y_var.shape else Y_mean
        prune_cutoff = np.partition(Y_ub, -l)[-l]
        idxs = np.arange(len(Y_mean))[Y_ub >= prune_cutoff]

        return idxs
    
    @staticmethod
    def prune_prob(
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        l: Union[int, float],
        min_hit_prob: float = 0.025,
    ) -> np.ndarray:
        """Prune all predictions with a probabilty less than min_hit_prob of being above a given
        threshold

        Parameters
        ----------
        Y_mean : np.ndarray
            the predicted means
        Y_var : np.ndarray
            the predicted variances
        l : Union[int, float]
            the percentile or rank of the predicted means from which to calculate a pruning 
            threshold 
        min_hit_prob : float
            the minimum probability necessary to avoid pruning

        Returns
        -------
        np.ndarray
            the indices of the predictions to retain
        """
        if isinstance(l, float):
            l = int(l * len(Y_mean))
        if l < 1:
            raise ValueError(f"l must be positive! got: {l}")
            
        prune_cutoff = np.partition(Y_mean, -l)[-l]
        P = MoleculePool.prob_above(Y_mean, Y_var, prune_cutoff)
        idxs = np.arange(len(Y_mean))[P >= min_hit_prob]

        return idxs

    @staticmethod
    def prune_max_fp(
        k: int,
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        max_fp: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        if isinstance(max_fp, float):
            max_fp *= len(Y_mean)
        if max_fp < 1:
            raise ValueError(f"max_fp must be positive! got: {max_fp}")

        sorted_idxs = np.argsort(Y_mean)[::-1]
        Y_mean = Y_mean[sorted_idxs]
        Y_var = Y_var[sorted_idxs]

        l = MoleculePool.maximize_fp(k, max_fp, Y_mean, Y_var)
        
        return sorted_idxs[:l]

    @staticmethod
    def expected_positives_pruned(
        k: int,
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        idxs: np.ndarray
    ) -> float:
        """the number of expected positives that will be pruned

        Parameters
        ----------
        k : int
            the rank above which to classify molecules as hits
        Y_mean : np.ndarray
            the predicted means
        Y_var : np.ndarray
            the predicted variances
        idxs : np.ndarray
            the molecules to retain after pruning

        Returns
        -------
        float
            the expected number of positives that will be pruned
        """
        hit_cutoff = np.partition(Y_mean, -k)[-k]

        if Y_mean.shape == Y_var.shape:
            P = MoleculePool.prob_above(Y_mean, Y_var, hit_cutoff)
        else:
            P = Y_mean >= hit_cutoff

        mask = np.zeros(len(Y_mean), bool)
        mask[idxs] = True

        return P[~mask].sum()

    @staticmethod
    def prob_above(Y_mean: np.ndarray, Y_var: np.ndarray, threshold: float) -> np.ndarray:
        """the probability that each prediction is above the input threshold

        Parameters
        ----------
        Y_mean : np.ndarray
            the mean of each prediction
        Y_var : np.ndarray
            the variance of each prediction
        cutoff : float
            the cutoff value

        Returns
        -------
        np.ndarray
        """
        I = Y_mean - threshold
        with np.errstate(divide='ignore'):
            Z = I / np.sqrt(Y_var)

        return norm.cdf(Z)

    @staticmethod
    def maximize_fp(k: int, max_fp: float, Y_mean: np.ndarray, Y_var: np.ndarray) -> int:
        """Return the number of molecules to select beyond the top-k such that the expected number 
        of false positives is maximized such that it is lower than an input threshold.
                
        Parameters
        ----------
        k : int
            the threshold above which to classify a molecule as a "postive" or "hit"
        max_fp : float
            the maximum allowable number of expected true positives
        Y_mean : np.ndarray
            the predicted mean of each molecule in the pool **sorted** by predicted mean
        Y_var : np.ndarray
            the predicted uncertainty of each molecule in the pool **sorted**
            by the associated predicted mean
        
        Returns
        -------
        int
            the number of top-predicted molecules to select
        """
        lo, hi = k, len(Y_mean)

        while lo < hi:
            mid = int(lo + hi) // 2
            E_TP = MoleculePool.expected_TP(Y_mean, Y_var, k, mid)
            if E_TP > max_fp:
                hi = mid
            else:
                lo = mid + 1

        return mid-1

    @staticmethod
    def expected_TP(Y_mean, Y_var, k: int, l: int):
        """the expected number of true positives between the k-th and l-th best predictions, where
        a positive is defined as a prediction being among the k best"""
        I = Y_mean - Y_mean[k-1]
        with np.errstate(divide='ignore'):
            Z = I / np.sqrt(Y_var)

        return norm.cdf(Z[k:l]).sum()

@ray.remote
def _validate_smis(smis):
    RDLogger.DisableLog("rdApp.*")
    return [smi if Chem.MolFromSmiles(smi) else None for smi in smis]


def validate_smis(smis) -> Iterator[Optional[str]]:
    refs = [_validate_smis.remote(smis_batch) for smis_batch in batches(smis, 4096)]
    for ref in refs:
        batch = ray.get(ref)
        for smi in batch:
            yield smi


class EagerMoleculePool(MoleculePool):
    """Alias for a MoleculePool"""
