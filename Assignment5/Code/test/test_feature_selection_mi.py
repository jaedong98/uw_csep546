import unittest

from utils.data_loader import get_xy_train_raw
from utils.feature_selection_mi import extract_features_by_mi


class FeaturesByMutualInformation(unittest.TestCase):


    def test_extract_features_by_mi(self):

        N = 10
        xTrainRaw, yTrainRaw = get_xy_train_raw()
        features = extract_features_by_mi(xTrainRaw, yTrainRaw, N)
        prev = features[0][1]
        for f in features[1:]:
            self.assertTrue(f[1] < prev)
            prev = f[1]


if __name__ == '__main__':
    unittest.main()
