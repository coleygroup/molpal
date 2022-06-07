from enum import Enum
from itertools import islice
from typing import Iterable, Iterator, List


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    @classmethod
    def from_str(cls, s: str):
        return cls[s.replace("-", "_").upper()]


def batches(it: Iterable, size: int) -> Iterator[List]:
    """Batch an iterable into batches of the given size, with the final batch
    potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])
