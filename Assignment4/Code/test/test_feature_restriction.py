import random
import unittest

from utils.feature_restriction import restrict_features, select_random_indices


class TestFeatureRestriction(unittest.TestCase):

    def test_restrict_features(self):

        data = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        selected_data = restrict_features(data, [0])
        for s, d in zip(selected_data, data):
            self.assertTrue(s[0], d[0])

        selected_data = restrict_features(data, [0, 2])
        for s, d in zip(selected_data, data):
            self.assertTrue(len(s) == 2)
            d.pop(1)
            self.assertTrue(s, d)

    def test_select_random_indices(self):
        seed = 10
        random.seed(seed)
        num_to_select = 2
        expected = random.sample(range(0, 5), num_to_select)
        result = select_random_indices(6, num_to_select, seed)
        for e, r in zip(expected, result):
            self.assertTrue(e == r)


if __name__ == '__main__':
    unittest.main()
