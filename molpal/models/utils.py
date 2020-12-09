"""utility functions for the models module"""
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
                   ncpu: int = 0) -> np.ndarray:
    """Calculate the feature matrix of xs with the given featurization
    function"""
    if ncpu <= 1:
        X = [featurize(x) for x in tqdm(xs, desc='Featurizing', smoothing=0.)]
    else:
        with Pool(max_workers=ncpu) as pool:
            X = list(tqdm(pool.map(featurize, xs), desc='Featurizing'))
    
    return np.array(X)
