import string
import unittest

import numpy as np

from molpal.acquirer import Acquirer

class TestAcquirer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs = [
            'CN=C=O', 'O=Cc1ccc(O)c(OC)c1', 'CC(=O)NCCC1=CNc2c1cc(OC)cc2',
            'CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4', 'CN1CCC[C@H]1c2cccnc2',
            'CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C',
            'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5',
            'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1',
            'CC[C@H](O1)CC[C@@]12CCCO2', 'c1cccc1', 'CCCC', 'c1cocc1',
            'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2', 'COc(c1)cccc1C#N'
            'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N', 'NC(C)C(=O)O', 'CC(C)C(=O)O'
            'CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C' 'O1CCOCC1',
            'C=C', 'C=1CC1', 'CCC(=O)O', 'Cc1ccccc1', 'CCN(CC)CC', 'CC(=O)O',
        ]

        cls.y_means = np.arange(len(cls.xs), dtype=np.double)[::-1]
        cls.y_vars = np.zeros(len(cls.xs))

        cls.init_size = 5
        cls.batch_size = 5
        cls.acq = Acquirer(
            size=len(cls.xs), init_size=cls.init_size,
            batch_size=cls.batch_size, metric='greedy', epsilon=0., seed=0
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
                         set(self.xs[2*self.batch_size:3*self.batch_size]))

    def test_acquire_prune(self):
        """make sure pruning is working properly with sizing"""
        for b in range(1,6):
            acq = Acquirer(
                size=len(self.xs), init_size=self.init_size,
                batch_size=self.batch_size,
                b=b, prune_mode='random',
                metric='greedy', epsilon=0., seed=None
            )
            batch_xs = acq.acquire_batch(
                self.xs, self.y_means, self.y_vars, explored={}
            )
            self.assertEqual(len(batch_xs), self.batch_size)

    def test_acquire_prune_random(self):
        """random pruning generally should not yield the same points. This
        test has 1/4**2 = 6.25% chance of randomly failing and a ~0.4% chance
        of randomly failing twice in a row."""
        b=4
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=b, prune_mode='random',
            metric='greedy', epsilon=0., seed=None
        )
        batch_xs_1 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        batch_xs_2 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), len(batch_xs_2))
        self.assertNotEqual(set(batch_xs_1), set(batch_xs_2))
    
    def test_acquire_prune_random_b1(self):
        """the two sets should be equal because their initial candidate sets
        are of size batch_size"""
        b=1
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=b, prune_mode='random',
            metric='greedy', epsilon=0., seed=None
        )
        batch_xs_1 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        batch_xs_2 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), len(batch_xs_2))
        self.assertEqual(set(batch_xs_1), set(batch_xs_2))
    
    def test_acquire_prune_maxmin(self):
        """the two sets should be equal with the same random seed"""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=2, prune_mode='maxmin', prune_threshold=0.,
            metric='greedy', epsilon=0., seed=42
        )
        batch_xs_1 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), self.batch_size)

        batch_xs_2 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), len(batch_xs_2))
        self.assertEqual(set(batch_xs_1), set(batch_xs_2))

    def test_acquire_prune_leader_t000(self):
        """the two sets should be equal with the same random seed and distance
        threshold of 0"""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=2, prune_mode='leader', prune_threshold=0.00,
            metric='greedy', epsilon=0., seed=42
        )
        batch_xs_1 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), self.batch_size)

        batch_xs_2 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(len(batch_xs_1), len(batch_xs_2))
        self.assertEqual(set(batch_xs_1), set(batch_xs_2))

    def test_acquire_prune_leader_t035(self):
        """leader picking is deterministic so the two batches should be equal"""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=3, prune_mode='leader', prune_threshold=0.35,
            metric='greedy', epsilon=0.
        )
        batch_xs_1 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        batch_xs_2 = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertEqual(batch_xs_1, batch_xs_2)

    def test_acquire_prune_leader_t100(self):
        """the acquirer shouldn't be able to find enough candidates that are
        diverse enough to pick batch_size candidates"""
        acq = Acquirer(
            size=len(self.xs), init_size=self.init_size,
            batch_size=self.batch_size,
            b=3, prune_mode='leader', prune_threshold=1.00,
            metric='greedy', epsilon=0.
        )
        batch_xs = acq.acquire_batch(
            self.xs, self.y_means, self.y_vars, explored={}
        )
        self.assertLess(len(batch_xs), self.batch_size)

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