import numpy as np
import pytest

from molpal.acquirer import metrics

@pytest.fixture(params=[0, 1, 2, 4])
def beta(request):
    return request.param

@pytest.fixture(params=[-1, 0, 1000])
def curr_max(request):
    return request.param

@pytest.fixture(params=[-1, 0, 5, 1000])
def threshold(request):
    return request.param

@pytest.fixture(params=[0., 0.01, 0.1])
def xi(request):
    return request.param

@pytest.fixture(params=[True, False])
def stochastic(request):
    return request.param

@pytest.fixture(params=[10, 50, 100])
def size(request):
    return request.param

@pytest.fixture
def Y_mean(size: int):
    return np.arange(size)

@pytest.fixture
def Y_var_0(size: int):
    return np.zeros(size)

@pytest.fixture
def Y_var(size: int):
    return np.random.rand(size) / 10

def test_random(Y_mean):
    U = metrics.random(Y_mean)

    np.testing.assert_equal(Y_mean.shape, U.shape)

def test_greedy(Y_mean):
    U = metrics.greedy(Y_mean)

    np.testing.assert_array_equal(Y_mean, U)

def test_noisy(Y_mean):
    sd = np.std(Y_mean)
    U = metrics.noisy(Y_mean)

    np.testing.assert_allclose(Y_mean, U, atol=4*sd)
        
def test_ucb(Y_mean, Y_var, beta):
    U = metrics.ucb(Y_mean, Y_var, beta)

    np.testing.assert_allclose(U, Y_mean + beta*np.sqrt(Y_var))

def test_ts_no_var(Y_mean):
    Y_var = np.zeros(Y_mean.shape)
    U = metrics.thompson(Y_mean, Y_var)
    np.testing.assert_allclose(U, Y_mean)

def test_ts_tiny_var(Y_mean):
    Y_var = np.full(Y_mean.shape, 0.01)
    U = metrics.thompson(Y_mean, Y_var)
    np.testing.assert_allclose(U, Y_mean, atol=0.4)

def test_thompson(Y_mean, Y_var, stochastic):
    Y_var = np.zeros(Y_mean.shape)
    U = metrics.thompson(Y_mean, Y_var, stochastic)

    if stochastic:
        np.testing.assert_equal(Y_mean, U)
    else:
        np.testing.assert_allclose(U, Y_mean, atol=0.4)

def test_ei(Y_mean: np.ndarray, Y_var_0: np.ndarray, xi, curr_max):
    U = metrics.ei(Y_mean, Y_var_0, curr_max, xi)

    if curr_max > Y_mean.max():
        np.testing.assert_array_less(U, 0)
    elif curr_max < Y_mean.min():
        np.testing.assert_array_less(0, U)
    else:
        np.testing.assert_array_less(-1, U)

# def test_pi_no_imp(Y_mean: np.ndarray, Y_var: np.ndarray, xi: float, curr_max: float):
#     Y_var = np.zeros(Y_mean.shape)
#     xi = 0.
#     curr_max = 100

#     U = metrics.pi(Y_mean, Y_var, current_max=curr_max, xi=xi)
#     np.testing.assert_array_equal(U, np.zeros(Y_mean.shape))

# def test_pi_with_imp(Y_mean: np.ndarray, Y_var: np.ndarray):
#     Y_var_1 = np.zeros(Y_mean.shape)
#     xi = 0.
#     curr_max = -1.

#     U = metrics.pi(Y_mean, Y_var_1, current_max=curr_max, xi=xi)
#     np.testing.assert_array_less(np.zeros(Y_mean.shape), U)

def test_threshold(Y_mean: np.ndarray, threshold: float):
    U = metrics.threshold(Y_mean, threshold)

    np.testing.assert_array_less(0, U[Y_mean >= threshold])
    np.testing.assert_array_equal(-1, U[Y_mean < threshold])
