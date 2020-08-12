from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from itertools import islice
from typing import Callable, Iterable, Iterator, List, Type, TypeVar

import numpy as np
from tqdm import tqdm

from ..encoders import Encoder, AtomPairFingerprinter
T = TypeVar('T')

# TODO: allow model class that tries many options and chooses
# best architecture when called for the very first time based
# on the first training batch

# TODO: allow classification tasks

def batches(it: Iterable[T], chunk_size: int) -> Iterator[List]:
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def get_model_types() -> List[str]:
    return ['rf', 'gp', 'nn', 'mpn']

def feature_matrix(xs: Iterable[T], featurize: Callable[[T], np.ndarray],
                   n_workers: int = 0) -> np.ndarray:
    if n_workers <= 1:
        return np.stack([featurize(x) for x in xs])    

    global _featurize; _featurize = featurize
    with Pool(max_workers=n_workers) as pool:
        X = list(pool.map(__featurize, xs))
        return np.stack(X)

# weird global definitions are necessary to allow for pickling and
# parallelization of feature_matrix generation
_featurize_ = None
def __featurize(x):
    return _featurize(x)
