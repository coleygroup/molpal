import csv
from itertools import chain, islice
from timeit import default_timer
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TypeVar

import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
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

def parse_csv(score_csv: str, title_line: bool = True,
              smiles_col: int = 0, score_col: int = 1,
              k: Optional[int] = None) -> Dict[str, Optional[float]]:
    with open(score_csv, 'r') as fid:
        reader = csv.reader(fid)

        n_rows = sum(1 for _ in reader); fid.seek(0)
        if title_line:
            next(reader)
            n_rows -= 1

        k = k or n_rows
        data = {
            row[smiles_col]: -float(row[score_col]) if row[score_col] else None
            for row in islice(reader, k)
        }
    
    return data

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
