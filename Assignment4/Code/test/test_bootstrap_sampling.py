import random
import unittest

from utils.bootstrap_sampling import get_bagging_indices


class TestBootstraping(unittest.TestCase):

    def test_get_bagging_indices(self):

        seed = 10
        sample_size = 10
        random.seed(seed)
        expected = [random.randint(0, sample_size - 1)
                    for _ in range(sample_size)]
        result = get_bagging_indices(sample_size, seed)

        for e, r in zip(expected, result):
            self.assertTrue(e == r, '{} vs {}'.format(e, r))
            self.assertTrue(r < sample_size)
