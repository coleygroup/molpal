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
def min_hit_prob(request):
    return request.param

def test_expected_pos_pruned_no_var_no_pruning(k, Y_mean, Y_var):
    k = int(k * len(Y_mean))

    E_pos_pruned = MoleculePool.expected_positives_pruned(k, Y_mean, Y_var, np.arange(len(Y_mean)))

    assert E_pos_pruned == 0

def test_expected_pos_pruned_no_var_all_pruned(k, Y_mean):
    k = int(k * len(Y_mean))
    hit_cutoff = np.partition(Y_mean, -k)[-k]

    E_pos_pruned = MoleculePool.expected_positives_pruned(k, Y_mean, np.array([]), [])

    assert E_pos_pruned == (Y_mean >= hit_cutoff).sum()

def test_expected_pos_pruned_no_var_all_hits(Y_mean, idxs):
    E_pos_pruned = MoleculePool.expected_positives_pruned(0, Y_mean, np.array([]), idxs)

    assert E_pos_pruned == len(Y_mean) - len(idxs)

def test_expected_pos_pruned_no_var_single_hit(Y_mean, idxs):
    E_pos_pruned = MoleculePool.expected_positives_pruned(1, Y_mean, np.array([]), idxs)

    assert E_pos_pruned == 1 or E_pos_pruned == 0

def test_prob_above(Y_mean, Y_var):
    P = MoleculePool.prob_above(Y_mean, Y_var, Y_mean.max())

    assert np.all(P <= 0.5)

def test_maximize_fp(k, max_fp, Y_mean, Y_var):
    k = int(k * len(Y_mean))
    max_fp *= len(Y_mean)

    sorted_idxs = np.argsort(Y_mean)[::-1]
    Y_mean = Y_mean[sorted_idxs]
    Y_var = Y_var[sorted_idxs]

    l = MoleculePool.maximize_fp(k, max_fp, Y_mean, Y_var)
    assert MoleculePool.expected_TP(Y_mean, Y_var, k, l) <= max_fp

def test_prune_greedy(Y_mean, l):
    l = int(l * len(Y_mean))

    idxs = MoleculePool.prune_greedy(Y_mean, l)

    assert len(idxs) == l

def test_prune_ucb(Y_mean, Y_var, l, beta):
    l = int(l * len(Y_mean))
    idxs = MoleculePool.prune_ucb(Y_mean, Y_var, l, beta)

    Y_ub = Y_mean + beta*np.sqrt(Y_var)
    prune_cutoff = np.partition(Y_mean, -l)[-l]

    assert len(idxs) == l
    assert np.all(Y_ub[idxs] >= prune_cutoff)

def test_prune_prob(Y_mean, Y_var, l, min_hit_prob):
    l = int(l * len(Y_mean))
    idxs = MoleculePool.prune_prob(Y_mean, Y_var, l, min_hit_prob)

    prune_cutoff = np.partition(Y_mean, -l)[-l]
    P = MoleculePool.prob_above(Y_mean, Y_var, prune_cutoff)

    assert np.all(P[idxs] >= min_hit_prob)