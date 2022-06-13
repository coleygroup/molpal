from random import random
import string
import uuid

import numpy as np
import pytest

from molpal.acquirer import Acquirer


@pytest.fixture(
    params=[list(string.printable), list(range(50)), list(str(uuid.uuid4()) + str(uuid.uuid4()))]
)
def xs(request):
    return request.param


@pytest.fixture(params=2 ** np.arange(4))
def explored(xs, request):
    idxs = np.random.choice(len(xs), request.param)

    return {xs[i]: random() for i in idxs}


@pytest.fixture
def Y_mean(xs):
    return np.random.normal(len(xs), size=len(xs))


@pytest.fixture
def Y_var(xs):
    return np.random.normal(size=len(xs)) / 10


@pytest.fixture
def init_size():
    return 10


@pytest.fixture
def batch_sizes():
    return [10]


@pytest.fixture(params=[None, 42])
def seed(request):
    return request.param


@pytest.fixture
def acq(xs, init_size, batch_sizes, seed):
    return Acquirer(len(xs), init_size, batch_sizes, "greedy", seed=seed)


def test_acquire_initial(acq, xs):
    xs_0 = acq.acquire_initial(xs)

    assert len(xs_0) == acq.init_size


def test_acquire_initial_reacquire(acq, xs):
    xs_0 = acq.acquire_initial(xs)
    xs_1 = acq.acquire_initial(xs)

    assert xs_0 != xs_1


def test_acquire_initial_reacquire_seeded(acq, xs, seed):
    if seed is None:
        return

    xs_0 = acq.acquire_initial(xs)
    acq.reset()
    xs_1 = acq.acquire_initial(xs)

    assert xs_0 == xs_1


def test_acquire_batch_size(acq, xs, Y_mean, Y_var, explored):
    batch = acq.acquire_batch(xs, Y_mean, Y_var, explored)

    assert len(batch) == acq.batch_sizes[0]


def test_acquire_batch_unique(acq, xs, Y_mean, Y_var, explored):
    batch = acq.acquire_batch(xs, Y_mean, Y_var, explored)

    assert all(x not in explored for x in batch)


@pytest.mark.parametrize("k", [0, 0.5, 2])
def test_acquire_batch_k(acq: Acquirer, xs, Y_mean, Y_var, explored, k):
    k = min(1, int(k * len(explored)))
    batch = acq.acquire_batch(xs, Y_mean, Y_var, explored, k)

    assert all(x not in explored for x in batch)


def test_acquire_batch_top_m(acq, xs, Y_mean, Y_var):
    batch = acq.acquire_batch(xs, Y_mean, Y_var, None)

    top_m_idxs = np.argsort(Y_mean)[: -1 - acq.batch_sizes[0] : -1]
    top_m_xs = [xs[i] for i in top_m_idxs]

    assert all(x in top_m_xs for x in batch)


def test_acquire_batch_determinism(acq, xs, Y_mean, Y_var):
    acq.epsilon = 0

    batch_1 = acq.acquire_batch(xs, Y_mean, Y_var, None)
    batch_2 = acq.acquire_batch(xs, Y_mean, Y_var, None)

    assert batch_1 == batch_2


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0])
def test_aquire_batch_nondetermism(acq, xs, Y_mean, Y_var, epsilon):
    """this test may randomly fail but the chances of that are low. repeated failures indicate
    problems"""
    acq.epsilon = epsilon
    batch_1 = acq.acquire_batch(xs, Y_mean, Y_var, None)
    batch_2 = acq.acquire_batch(xs, Y_mean, Y_var, None)

    top_m_idxs = np.argsort(Y_mean)[: -1 - acq.batch_sizes[0] : -1]
    top_m_xs = np.array(xs)[top_m_idxs]

    assert len(batch_1) == len(batch_2) == len(top_m_xs)

    if epsilon == 0.0:
        assert batch_1 == batch_2
    else:
        assert batch_1 != batch_2
