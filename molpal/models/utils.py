"""This module contains utility functions for the models module"""

from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import islice
from typing import Callable, Iterable, Iterator, List, TypeVar

import numpy as np
from tqdm import tqdm

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
    """Calculate the feature matrix of xs with the given featurization
    function"""
    if n_workers <= 1:
        X = [featurize(x) for x in tqdm(xs, desc='Featurizing', smoothing=0.)]
    else:
        with Pool(max_workers=n_workers) as pool:
            # global featurize_; featurize_ = featurize
            X = list(tqdm(pool.map(featurize, xs), desc='Featurizing'))
    
    return np.array(X)

# local featurize function isn't pickleable so it must be wrapped inside a
# global function to enable parallel feature matrix construction
# def __featurize(x):
#     return featurize_(x)
