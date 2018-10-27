import unittest

from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


class TestRandomForestModel(unittest.TestCase):

    def test_fit_predicts(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295)
        for fr in [0, 10, 50, 100, 200, 300]:
            rfm = RandomForestModel(numTrees=20,
                                    use_bagging=True,
                                    feature_restriction=fr,
                                    seed=100)
            rfm.fit(xTrain, yTrain, min_to_split=20)
            yTestPredicted = rfm.predict(xTest)
            print(Evaluation(yTest, yTestPredicted))


if __name__ == '__main__':
    unittest.main()
