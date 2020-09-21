import string
import unittest

import numpy as np

from molpal.acquirer import Acquirer

class TestAcquirer(unittest.TestCase):
    def setUp(self):
        self.xs = string.ascii_lowercase
        self.y_means = np.arange(len(self.xs))[::-1]
        self.y_vars = np.zeros(len(self.xs))

        self.init_size = 10
        self.batch_size = 10
        self.acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=0., seed=0
        )

    def test_acquire_initial(self):
        init_xs = self.acq.acquire_initial(self.xs)
        self.assertEqual(len(init_xs), self.init_size)

    def test_acquire_initial_set_seed(self):
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=0., seed=0
        )
        init_xs_1 = acq.acquire_initial(self.xs)

        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=0., seed=0
        )
        init_xs_2 = acq.acquire_initial(self.xs)

        self.assertEqual(set(init_xs_1), set(init_xs_2))
    
    def test_acquire_initial_None_seed(self):
        """There is a roughly 1-in-2.5E13 chance that the same initial batch
        is chosen twice in a row, causing this test to report failure. If this
        test fails twice in a row, it is safe to assume that the test is 
        genuinely failing."""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=0., seed=None
        )
        init_xs_1 = acq.acquire_initial(self.xs)

        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=0., seed=None
        )
        init_xs_2 = acq.acquire_initial(self.xs)

        self.assertNotEqual(set(init_xs_1), set(init_xs_2))

    def test_acquire_batch_explored(self):
        """Acquirers should not reacquire old points"""
        init_xs = self.acq.acquire_initial(self.xs)
        explored = {x: 0. for x in init_xs}

        batch_xs = self.acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored=explored
        )
        self.assertEqual(len(batch_xs), self.batch_size)
        self.assertNotEqual(set(init_xs), set(batch_xs))
    
    def test_acquire_batch_correct_points(self):
        """Acquirers should acquire the top-m points by calculated utility.
        This test acquirer uses the greedy metric because the utilities are
        equal to the input Y_means. Assuming utility calculation is working
        correctly, this test ensures that the acquirer uses these utilities
        properly."""
        batch_xs = self.acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(set(batch_xs),
                         set(self.xs[:self.batch_size]))

        explored = {x: 0. for x in batch_xs}
        batch_xs_2 = self.acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored=explored
        )
        self.assertEqual(set(batch_xs_2),
                         set(self.xs[self.batch_size:2*self.batch_size]))

        explored_2 = {x: 0. for x in batch_xs_2}
        batch_xs_3 = self.acq.acquire_batch(
            self.xs, self.y_means, self.y_vars,
            explored={**explored, **explored_2}
        )
        self.assertEqual(set(batch_xs_3),
                         set(self.xs[2*self.batch_size:]))

    def test_acquire_batch_epsilon(self):
        """There is roughly a  1-in-5*10^6 (= nCr(26, 10)) chance that a random
        batch is the same as the calculated top-m batch, causing this test to
        report a failure. There is a roughly 1-in-2.5E13 chance this test
        fails twice in arow. If that happens, it is safe to assume that the
        test is genuinely failing."""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size, metric='greedy', epsilon=1., seed=0
        )
        explored = {}

        batch_xs = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored=explored
        )
        self.assertNotEqual(set(batch_xs),
                         set(self.xs[:self.batch_size]))

if __name__ == "__main__":
    unittest.main()