import unittest

from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


class TestRandomForestModel(unittest.TestCase):

    def test_accuracy_with_feature_extraction_w_bagging(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295)
        for fr in [10, 50, 100, 200, 300]:
            rfm = RandomForestModel(numTrees=20,
                                    use_bagging=True,
                                    feature_restriction=fr,
                                    seed=100)
            rfm.fit(xTrain, yTrain, min_to_split=100)
            yTestPredicted = rfm.predict(xTest)
            print(Evaluation(yTest, yTestPredicted))

    def test_accuracy_with_feature_extraction_wo_bagging(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295)
        for fr in [10, 50, 100, 200, 300]:
            rfm = RandomForestModel(numTrees=20,
                                    use_bagging=True,
                                    feature_restriction=fr,
                                    seed=100)
            rfm.fit(xTrain, yTrain, min_to_split=100)
            yTestPredicted = rfm.predict(xTest)
            print(Evaluation(yTest, yTestPredicted))

    def test_accuracy_baseline_wo_noise(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=False)

        rfm = RandomForestModel(numTrees=10,
                                use_bagging=False,
                                feature_restriction=20,
                                seed=100)
        rfm.fit(xTrain, yTrain, min_to_split=100)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))

    def test_accuracy_baseline_w_noise(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=True)

        rfm = RandomForestModel(numTrees=10,
                                use_bagging=False,
                                feature_restriction=20,
                                seed=100)
        rfm.fit(xTrain, yTrain, min_to_split=2)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))


if __name__ == '__main__':
    unittest.main()
