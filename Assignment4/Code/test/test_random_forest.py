import unittest

from model.RandomForestsModel import RandomForestModel
from utils.data_loader import get_featurized_xs_ys


class TestRandomForestModel(unittest.TestCase):

    def test_fit(self):
        rfm = RandomForestModel(1, use_bagging=False, feature_restriction=0, seed=0)
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys()
        rfm.fit(xTrain, yTrain, min_to_split=2)


if __name__ == '__main__':
    unittest.main()
