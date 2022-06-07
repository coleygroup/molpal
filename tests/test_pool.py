import numpy as np
import pytest

from molpal.pools.base import MoleculePool

np.random.seed(42)

@pytest.fixture(params=[1000*(10**i) for i in range(3)])
def size(request):
    return request.param

@pytest.fixture(params=[0.01, 0.1, 0.5])
def idxs(request, size):
    return np.random.choice(size, int(size*request.param), replace=False)

@pytest.fixture
def Y_mean(size):
    return np.random.normal(8, 1.5, size)

@pytest.fixture
def Y_var(size):
    return np.random.uniform(0, 1, size)

@pytest.fixture(params=[0.0005, 0.001, 0.005, 0.01])
def k(request):
    return request.param

@pytest.fixture(params=[0.01, 0.05, 0.1])
def l(request):
    return request.param

@pytest.fixture(params=[1., 2.])
def beta(request):
    return request.param

@pytest.fixture(params=[0.01, 0.05, 0.1])
def max_fp(request):
    return request.param

@pytest.fixture(params=[0.025, 0.01, 0.001])
def p_min(request):
    return request.param

@pytest.fixture(params=[0.05, 0.10])
def max_pos_prune(request):
    return request.param

def test_expected_pos_pruned_no_var_no_pruning(k, Y_mean, Y_var):
    k = int(k * len(Y_mean))
    threshold = np.partition(Y_mean, -k)[-k]

    E_pos_pruned = MoleculePool.expected_positives_pruned(
        threshold, Y_mean, Y_var, np.arange(len(Y_mean))
    )

    assert E_pos_pruned == 0

def test_expected_pos_pruned_no_var_all_pruned(k, Y_mean):
    k = int(k * len(Y_mean))
    threshold = np.partition(Y_mean, -k)[-k]

    E_pos_pruned = MoleculePool.expected_positives_pruned(threshold, Y_mean, np.array([]), [])

    assert E_pos_pruned == (Y_mean >= threshold).sum()

def test_expected_pos_pruned_no_var_all_hits(Y_mean, idxs):
    threshold = Y_mean.min()
    E_pos_pruned = MoleculePool.expected_positives_pruned(threshold, Y_mean, np.array([]), idxs)

    assert E_pos_pruned == len(Y_mean) - len(idxs)

def test_expected_pos_pruned_no_var_single_hit(Y_mean, idxs):
    threshold = Y_mean.max()
    E_pos_pruned = MoleculePool.expected_positives_pruned(threshold, Y_mean, np.array([]), idxs)

    assert E_pos_pruned == 1 or E_pos_pruned == 0

def test_prob_above(Y_mean, Y_var):
    P = MoleculePool.prob_above(Y_mean, Y_var, Y_mean.max())

    assert np.all(P <= 0.5)

def test_prune_prob(Y_mean, Y_var, l, p_min):
    l = int(l * len(Y_mean))
    threshold = np.partition(Y_mean, -l)[-l]

    idxs = MoleculePool.prune_prob(threshold, Y_mean, Y_var, p_min)
    P = MoleculePool.prob_above(Y_mean, Y_var, threshold)

    assert np.all(P[idxs] >= p_min)