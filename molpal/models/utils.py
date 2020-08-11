"""This module contains the Model ABC and various implementations thereof. A
model is used to predict an input's objective function based on prior
training data."""

from itertools import islice
from typing import Iterable, Iterator, List, TypeVar

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