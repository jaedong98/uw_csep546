import unittest

from utils.feature_restriction import restrict_features


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


if __name__ == '__main__':
    unittest.main()
