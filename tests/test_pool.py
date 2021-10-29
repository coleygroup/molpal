import numpy as np
import pytest
from scipy.stats import norm

from molpal.pools.base import MoleculePool

np.random.seed(42)

@pytest.fixture(params=[0.0005, 0.001, 0.005, 0.01])
def k(request):
    return request.param

@pytest.fixture(params=[0.01, 0.05, 0.1])
def max_fp(request):
    return request.param

@pytest.fixture(params=[1000*(10**i) for i in range(3)])
def size(request):
    return request.param

@pytest.fixture
def Y_mean(size):
    return np.random.normal(8, 1.5, size)

@pytest.fixture
def Y_var(size):
    return np.ones(size)

def test_prune(k, max_fp, Y_mean, Y_var):
    k = int(k * len(Y_mean))
    max_fp *= len(Y_mean)

    sorted_idxs = np.argsort(Y_mean)[::-1]
    Y_mean = Y_mean[sorted_idxs]
    Y_var = Y_var[sorted_idxs]

    l = MoleculePool.maximize_fp(k, max_fp, Y_mean, Y_var)
    assert MoleculePool.expected_FP(Y_mean, Y_var, k, l) <= max_fp
