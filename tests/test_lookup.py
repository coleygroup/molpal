import csv
import random
import string
from typing import Iterable, List
import uuid

import numpy as np
import pytest

from molpal.objectives.lookup import LookupObjective

def interleave(a: Iterable, b: Iterable) -> List:
    return list(map(next, random.sample([iter(a)]*len(a) + [iter(b)]*len(b), len(a)+len(b))))

@pytest.fixture
def empty(tmp_path) -> LookupObjective:
    p_config = tmp_path / "config.ini"
    p_csv = tmp_path / "obj.csv"

    with open(p_config, "w") as fid:
        fid.write(f"path: {p_csv}\n")
        fid.write(f"no-title-line: True\n")

    with open(p_csv, "w") as fid:
        pass

    return LookupObjective(str(p_config), minimize=False)

@pytest.fixture
def singleton(tmp_path) -> LookupObjective:
    p_config = tmp_path / "config.ini"
    p_csv = tmp_path / "obj.csv"

    with open(p_config, "w") as fid:
        fid.write(f"path: {p_csv}\n")
        fid.write(f"no-title-line: True\n")

    with open(p_csv, "w") as fid:
        writer = csv.writer(fid)
        writer.writerow(["foo", "42"])

    return LookupObjective(str(p_config), minimize=False)

@pytest.fixture
def normal(tmp_path, normal_xs, normal_ys) -> LookupObjective:
    p_config = tmp_path / "config.ini"
    p_csv = tmp_path / "obj.csv"

    with open(p_config, "w") as fid:
        fid.write(f"path: {p_csv}\n")
        fid.write(f"no-title-line: True\n")

    with open(p_csv, "w") as fid:
        writer = csv.writer(fid)
        writer.writerows(zip(normal_xs, normal_ys))

    return LookupObjective(str(p_config), minimize=False)

@pytest.fixture
def normal_min(tmp_path, normal_xs, normal_ys) -> LookupObjective:
    p_config = tmp_path / "config.ini"
    p_csv = tmp_path / "obj.csv"

    with open(p_config, "w") as fid:
        fid.write(f"path: {p_csv}\n")
        fid.write(f"no-title-line: True\n")

    with open(p_csv, "w") as fid:
        writer = csv.writer(fid)
        writer.writerows(zip(normal_xs, normal_ys))

    return LookupObjective(str(p_config))

@pytest.fixture
def sparse(tmp_path, sparse_xs, sparse_ys) -> LookupObjective:
    p_config = tmp_path / "config.ini"
    p_csv = tmp_path / "obj.csv"

    with open(p_config, "w") as fid:
        fid.write(f"path: {p_csv}\n")
        fid.write(f"no-title-line: True\n")
        fid.write(f"smiles-col: 1\n")
        fid.write(f"score-col: 4\n")

    with open(p_csv, "w") as fid:
        writer = csv.writer(fid)

        for x, y in zip(sparse_xs, sparse_ys):
                writer.writerow([None, x, None, None, y])

    return LookupObjective(str(p_config), minimize=False)

@pytest.fixture(params=[string.ascii_letters, [str(uuid.uuid4()) for _ in range(100)]])
def normal_xs():
    return string.ascii_lowercase

@pytest.fixture
def normal_ys(normal_xs):
    return np.random.randn(len(normal_xs))

@pytest.fixture(params=[2, 5, 10])
def xs(request, normal_xs):
    return random.sample(normal_xs, request.param)

@pytest.fixture
def sparse_xs():
    return ["foo", "bar", "alice", "bob"]

@pytest.fixture
def sparse_ys(sparse_xs):
    return np.random.uniform(0, 100, len(sparse_xs))

def test_forward_and_call(normal, xs):
    ys1 = np.array(list(normal.forward(xs).values()))
    ys2 = np.array(list(normal(xs).values()))

    assert len(ys1) == len(ys2)
    np.testing.assert_allclose(ys1, ys2)
 
def test_empty(empty, xs):
    ys = empty(xs).values()

    assert all([y is None for y in ys])

def test_singleton_integrity(singleton):
    np.testing.assert_almost_equal(singleton(["foo"])["foo"], 42)

def test_singleton_not_contained(singleton, xs):
    ys = singleton(xs).values()

    assert all([y is None for y in ys])

def test_length(normal, xs):
    ys = normal(xs)

    assert len(ys) == len(xs)

def test_normal_integrity(normal, normal_xs, normal_ys):
    actual = list(normal(normal_xs).values())

    np.testing.assert_allclose(actual, normal_ys)

def test_minimization(normal_min, normal_xs, normal_ys):
    actual = list(normal_min(normal_xs).values())

    np.testing.assert_allclose(actual, -normal_ys)

def test_normal_some_contained(normal, normal_xs, sparse_xs):
    xs = interleave(normal_xs, sparse_xs)
    d_xy = normal(xs)

    normal_xs = set(normal_xs)
    for x, y in d_xy.items():
        if x not in normal_xs:
            assert y is None

def test_sparse(sparse, sparse_xs, sparse_ys):
    ys = list(sparse(sparse_xs).values())

    np.testing.assert_allclose(ys, sparse_ys)