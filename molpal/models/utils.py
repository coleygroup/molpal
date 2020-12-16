"""utility functions for the models module"""
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from itertools import islice
import os
from typing import Callable, Iterable, Iterator, List, TypeVar

import numpy as np
from tqdm import tqdm

T = TypeVar('T')

try:
    MAX_CPU = len(os.sched_getaffinity(0))
except AttributeError:
    MAX_CPU = os.cpu_count()

def batches(it: Iterable[T], chunk_size: int) -> Iterator[List]:
    """Batch an iterable into batches of size chunk_size, with the final
    batch potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def get_model_types() -> List[str]:
    return ['rf', 'gp', 'nn', 'mpn']

def feature_matrix(xs: Iterable[T], featurize: Callable[[T], np.ndarray],
                   num_workers: int = 1, ncpu: int = 1,
                   distributed: bool = False,
                   disable: bool = False) -> np.ndarray:
    """Calculate the feature matrix of xs with the given featurization
    function"""
    if distributed:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor as Pool

        comm = MPI.COMM_WORLD
        num_workers = comm.Get_size()

        if ncpu > 1 and num_workers > 1:
            feature_matrix_ = partial(
                pmap, f=featurize, num_workers=ncpu, disable=True
            )
            with Pool(max_workers=num_workers) as pool:
                X = pool.map(feature_matrix_, batches(xs, chunk_size=256))
                X = list(tqdm(X, desc='Featurizing', unit='batch'))
            return np.vstack(X)
        elif num_workers > ncpu:
            return dmap(xs, featurize, disable)
        else:
            return pmap(xs, featurize, ncpu, disable)

    if num_workers == -1:
        num_workers = MAX_CPU
    num_workers *= ncpu
    
    if num_workers <= 1:
        X = map(featurize, xs)
        X = list(tqdm(X, desc='Featurizing', smoothing = 0., disable=disable))
        return np.array(X)
    
    return pmap(xs, featurize, num_workers, disable)

def pmap(xs: Iterable[T], f: Callable[[T], np.ndarray],
         num_workers: int = 1, disable: bool = False) -> np.ndarray:
    with Pool(max_workers=num_workers) as pool:
        X = pool.map(f, xs)
        X = list(tqdm(X, smoothing = 0., disable=disable))
    return np.array(X)

def dmap(xs: Iterable[T], f: Callable[[T], np.ndarray],
         disable: bool = False) -> np.ndarray:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor as Pool

    num_workers = MPI.COMM_WORLD.size
    with Pool(max_workers=num_workers) as pool:
        X = pool.map(f, xs)
        X = list(tqdm(X, smoothing = 0., disable=disable))
    return np.array(X)