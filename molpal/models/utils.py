"""This module contains utility functions for the models module"""

from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import islice
from typing import Callable, Iterable, Iterator, List, TypeVar

import numpy as np
# from tqdm import tqdm

T = TypeVar('T')

def batches(it: Iterable[T], chunk_size: int) -> Iterator[List]:
    """Batch an iterable into batches of size chunk_size, with the final
    batch potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def get_model_types() -> List[str]:
    return ['rf', 'gp', 'nn', 'mpn']

def feature_matrix(xs: Iterable[T], featurize: Callable[[T], np.ndarray],
                   n_workers: int = 0) -> np.ndarray:
    """Calculate the fature matrix of xs with the given featurization
    function using parallel processing"""
    if n_workers <= 1:
        X = [featurize(x) for x in xs]
    else:
        global _featurize; _featurize = featurize
        with Pool(max_workers=n_workers) as pool:
            X = list(pool.map(__featurize, xs))
    
    return np.array(X)

# weird global definitions are necessary to allow for pickling and
# parallelization of feature_matrix generation
_featurize = None
def __featurize(x):
    return _featurize(x)
