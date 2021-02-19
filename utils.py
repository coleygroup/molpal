import csv
from itertools import chain, islice
import os
from timeit import default_timer
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TypeVar

import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

T = TypeVar('T')
U = TypeVar('U')

if not ray.is_initialized():
    try:
        ray.init('auto')
    except ConnectionError:
        ray.init(num_cpus=len(os.sched_getaffinity(0)))

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

@ray.remote
def smis_to_fps_(smis: List[str], radius: int = 2, length: int = 2048) -> List:
    return [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), radius, length, useChirality=True
        ) for smi in smis
    ]

def smis_to_fps(smis: List[str], radius: int = 2, length: int = 2048) -> List:
    chunksize = int(ray.cluster_resources()['CPU'] * 512)
    refs = [
        smis_to_fps_.remote(smis_chunk, radius, length)
        for smis_chunk in chunks(smis, chunksize)
    ]
    fps_chunks = [ray.get(r) for r in tqdm(refs, unit='smis chunk')]
    fps = list(chain(*fps_chunks))

    return fps
