import unittest

from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


class TestRandomForestModel(unittest.TestCase):

    def test_fit_predicts(self):
        rfm = RandomForestModel(numTrees=2,
                                use_bagging=False,
                                feature_restriction=0,
                                seed=0)
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys()
        rfm.fit(xTrain, yTrain, min_to_split=50)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))


if __name__ == '__main__':
    unittest.main()
