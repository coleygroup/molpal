import unittest

import numpy as np
import numpy.testing as np_test

from molpal.acquirer import metrics

class TestMetrics(unittest.TestCase):    
    def test_random(self):
        Y_mean = np.arange(10)

        U = metrics.random_metric(Y_mean)
        self.assertEqual(U.shape, Y_mean.shape)

    def test_greedy(self):
        Y_mean = np.arange(10)

        U = metrics.greedy(Y_mean)
        np_test.assert_array_equal(U, Y_mean)

    def test_ucb_no_var(self):
        Y_mean = np.arange(10)
        Y_var = np.zeros(Y_mean.shape)
        beta = 2
        
        U = metrics.ucb(Y_mean, Y_var, beta=beta)
        np_test.assert_equal(U, Y_mean)

    def test_ucb_with_var(self):
        Y_mean = np.arange(10)
        Y_var = np.ones(Y_mean.shape)
        beta = 2
        
        U = metrics.ucb(Y_mean, Y_var, beta=beta)
        np_test.assert_allclose(U, Y_mean+beta)

    def test_ts_no_var(self):
        Y_mean = np.arange(10)

        Y_var = np.zeros(Y_mean.shape)
        U = metrics.thompson(Y_mean, Y_var)
        np_test.assert_allclose(U, Y_mean)

    def test_ts_tiny_var(self):
        Y_mean = np.arange(10)

        Y_var = np.full(Y_mean.shape, 0.01)
        U = metrics.thompson(Y_mean, Y_var)
        np_test.assert_allclose(U, Y_mean, atol=0.4)

    def test_ts_stochastic(self):
        Y_mean = np.arange(10)

        Y_var = np.zeros(Y_mean.shape)
        U = metrics.thompson(Y_mean, Y_var, stochastic=True)
        np_test.assert_equal(U, Y_mean)

    def test_ei_no_imp(self):
        Y_mean = np.arange(10)
        Y_var = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max = 100

        U = metrics.ei(Y_mean, Y_var, current_max=curr_max, xi=xi)
        np_test.assert_array_less(U, np.zeros(Y_mean.shape))

    def test_ei_with_imp(self):
        Y_mean = np.arange(10)
        Y_var = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max = 0

        U_2 = metrics.ei(Y_mean, Y_var, current_max=curr_max, xi=xi)
        np_test.assert_array_less(np.full(Y_mean.shape, -1.), U_2)

    def test_pi_no_imp(self):
        Y_mean = np.arange(10)
        Y_var = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max = 100

        U = metrics.pi(Y_mean, Y_var, current_max=curr_max, xi=xi)
        np_test.assert_array_equal(U, np.zeros(Y_mean.shape))

    def test_pi_with_imp(self):
        Y_mean = np.arange(10)
        Y_var_1 = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max = -1.

        U = metrics.pi(Y_mean, Y_var_1, current_max=curr_max, xi=xi)
        np_test.assert_array_less(np.zeros(Y_mean.shape), U)

    def test_threshold_all_above(self):
        Y_mean = np.arange(10)
        U = metrics.random_threshold(Y_mean, threshold=0)
        np_test.assert_array_less(U, np.ones(Y_mean.shape))

    def test_threshold_some_above(self):
        Y_mean = np.arange(10)
        U = metrics.random_threshold(Y_mean, threshold=4)
        np_test.assert_allclose(U[:5], np.full(U[:5].shape, -1.))

    def test_threshold_all_below(self):
        Y_mean = np.arange(10)
        U = metrics.random_threshold(Y_mean, threshold=10)
        np_test.assert_allclose(U, np.full(Y_mean.shape, -1.))
        
if __name__ == "__main__":
    unittest.main()