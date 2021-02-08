from itertools import chain, islice
from timeit import default_timer
from typing import Any, Callable, Iterable, Iterator, List, TypeVar

import numpy as np
import ray
from tqdm import tqdm

T = TypeVar('T')
U = TypeVar('U')

class Timer:
    def __enter__(self):
        self.start = default_timer()
    def __exit__(self, type, value, traceback):
        self.stop = default_timer()
        print(self.stop - self.start)


def normalize(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X)) / np.std(X)

def chunks(it: Iterable[T], chunksize: int) -> Iterator[List[T]]:
    """Batch an iterable into batches of size chunksize, with the final
    batch potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunksize)), [])

def pmap(f: Callable, xs: Iterable[T], chunksize: int = 1,
         *args, **kwargs) -> List[U]:
    if not ray.is_initialized():
        ray.init()

    @ray.remote
    def f_(ys: Iterable, *args, **kwargs):
        return [f(y, *args, **kwargs) for y in ys]

    args = [ray.put(arg) for arg in args]
    kwargs = {k: ray.put(v) for k, v in kwargs.items()}

    chunk_refs = [
        f_.remote(xs_batch, *args, **kwargs)
        for xs_batch in tqdm(chunks(xs, chunksize), desc='Submitting')
    ]
    y_chunks = [ray.get(ref) for ref in tqdm(chunk_refs, desc='Mapping')]
    return list(chain(*y_chunks))
