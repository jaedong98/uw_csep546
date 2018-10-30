import unittest

from utils.Assignment4Support import FeaturizeExt
from utils.data_loader import get_xy_train_raw, get_xy_test_raw


class TestFeatureize(unittest.TestCase):

    def test_featurize_ext(self):
        xTrainRaw, yTrainRaw = get_xy_train_raw()
        xTestRaw, _ = get_xy_test_raw()
        xTrain, xTest = FeaturizeExt(xTrainRaw, yTrainRaw, xTestRaw,
                                     numMutualInformationWords=295)
        num_train_features = len(xTrain[0])
        num_test_features = len(xTest[0])
        self.assertTrue(num_train_features == num_test_features)
        self.assertTrue(num_train_features == 300)


if __name__ == '__main__':
    unittest.main()
