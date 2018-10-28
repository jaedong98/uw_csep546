import unittest

from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


class TestRandomForestModel(unittest.TestCase):

    def test_accuracy_with_feature_extraction_w_bagging(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295)
        for fr in [10, 50, 100, 200, 300]:
            rfm = RandomForestModel(numTrees=20,
                                    bagging_w_replacement=True,
                                    feature_restriction=fr,
                                    seed=100)
            rfm.fit(xTrain, yTrain, min_to_split=100)
            yTestPredicted = rfm.predict(xTest)
            print(Evaluation(yTest, yTestPredicted))

    def test_accuracy_with_feature_extraction_wo_bagging(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295)
        for fr in [10, 50, 100, 200, 300]:
            rfm = RandomForestModel(numTrees=20,
                                    bagging_w_replacement=True,
                                    feature_restriction=fr,
                                    seed=100)
            rfm.fit(xTrain, yTrain, min_to_split=100)
            yTestPredicted = rfm.predict(xTest)
            print(Evaluation(yTest, yTestPredicted))

    def test_accuracy_baseline_wo_noise(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=False)

        rfm = RandomForestModel(numTrees=10,
                                bagging_w_replacement=True,
                                feature_restriction=20,
                                seed=100)
        rfm.fit(xTrain, yTrain, min_to_split=2)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))
        for i, p in enumerate(rfm.predictions):
            ev = Evaluation(yTest, p)
            print("Tree {}: {}".format(i, ev.accuracy))

    def test_accuracy_baseline_wo_noise_no_bootstrap(self):
        """
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 180 | (FN) 22  |
        |    0     | (FP) 19  |(TN) 1173 |
        Accuracy: 0.9705882352941176
        Precision: 0.9045226130653267
        Recall: 0.8910891089108911
        FPR: 0.015939597315436243
        FNR: 0.10891089108910891
        :return:
        """
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=False)

        rfm = RandomForestModel(numTrees=10,
                                bagging_w_replacement=False,
                                feature_restriction=0,
                                seed=100)
        rfm.fit(xTrain, yTrain, min_to_split=2)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))
        for i, p in enumerate(rfm.predictions):
            ev = Evaluation(yTest, p)
            print("Tree {}: {}".format(i, ev.accuracy))

    def test_accuracy_baseline_wo_noise_w_bootstrap(self):
        """
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     |  (TP) 2  | (FN) 200 |
        |    0     |  (FP) 9  |(TN) 1183 |
        Accuracy: 0.8500717360114778
        Precision: 0.18181818181818182
        Recall: 0.009900990099009901
        FPR: 0.007550335570469799
        FNR: 0.9900990099009901
        :return:
        """
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=False)

        rfm = RandomForestModel(numTrees=10,
                                bagging_w_replacement=True,
                                feature_restriction=0,
                                seed=100)
        rfm.fit(xTrain, yTrain, min_to_split=2)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))
        for i, p in enumerate(rfm.predictions):
            ev = Evaluation(yTest, p)
            print("Tree {}: {}".format(i, ev.accuracy))

    def test_accuracy_baseline_w_noise(self):

        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=295,
                                                            with_noise=True)

        rfm = RandomForestModel(numTrees=10,
                                bagging_w_replacement=True,
                                feature_restriction=20,
                                seed=1000)
        rfm.fit(xTrain, yTrain, min_to_split=2)
        yTestPredicted = rfm.predict(xTest)
        print(Evaluation(yTest, yTestPredicted))
        for i, p in enumerate(rfm.predictions):
            ev = Evaluation(yTest, p)
            print("Tree {}: {}".format(i, ev.accuracy))


if __name__ == '__main__':
    unittest.main()
