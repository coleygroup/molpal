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
    