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

    def test_ucb(self):
        Y_mean = np.arange(10)
        beta = 2

        Y_var_1 = np.zeros(Y_mean.shape)
        U_1 = metrics.ucb(Y_mean, Y_var_1, beta=beta)
        np_test.assert_equal(U_1, Y_mean)

        Y_var_2 = np.ones(Y_mean.shape)
        U_2 = metrics.ucb(Y_mean, Y_var_2, beta=beta)
        np_test.assert_allclose(U_2, Y_mean+beta)

    def test_ts(self):
        Y_mean = np.arange(10)

        Y_var_1 = np.zeros(Y_mean.shape)
        U_1 = metrics.thompson(Y_mean, Y_var_1)
        np_test.assert_allclose(U_1, Y_mean)

        Y_var_2 = np.full(Y_mean.shape, 0.01)
        U_2 = metrics.thompson(Y_mean, Y_var_2)
        np_test.assert_allclose(U_2, Y_mean, atol=0.4)

        U_3 = metrics.thompson(Y_mean, Y_var_1, stochastic=True)
        np_test.assert_equal(U_3, Y_mean)

    def test_ei(self):
        Y_mean = np.arange(10)
        Y_var_1 = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max_1 = 100
        curr_max_2 = 0

        U_1 = metrics.ei(Y_mean, Y_var_1, current_max=curr_max_1, xi=xi)
        np_test.assert_array_less(U_1, np.zeros(Y_mean.shape))

        U_2 = metrics.ei(Y_mean, Y_var_1, current_max=curr_max_2, xi=xi)
        np_test.assert_array_less(np.full(Y_mean.shape, -1.), U_2)

    def test_pi(self):
        Y_mean = np.arange(10)
        Y_var_1 = np.zeros(Y_mean.shape)
        xi = 0.
        curr_max_1 = 100
        curr_max_2 = -1.

        U_1 = metrics.pi(Y_mean, Y_var_1, current_max=curr_max_1, xi=xi)
        np_test.assert_array_equal(U_1, np.zeros(Y_mean.shape))

        U_2 = metrics.pi(Y_mean, Y_var_1, current_max=curr_max_2, xi=xi)
        np_test.assert_array_less(np.zeros(Y_mean.shape), U_2)

    def test_threshold(self):
        Y_mean = np.arange(10)

        U_1 = metrics.random_threshold(Y_mean, threshold=0)
        np_test.assert_array_less(U_1, np.ones(Y_mean.shape))
        
        U_2 = metrics.random_threshold(Y_mean, threshold=4)
        np_test.assert_allclose(U_2[:5], np.full(U_2[:5].shape, -1.))
        
        U_3 = metrics.random_threshold(Y_mean, threshold=10)
        np_test.assert_allclose(U_3, np.full(Y_mean.shape, -1.))
        
if __name__ == "__main__":
    unittest.main()