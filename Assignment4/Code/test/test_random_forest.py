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
        * feature_restriction = 0 (with all features)
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 175 | (FN) 27  |
        |    0     | (FP) 31  |(TN) 1161 |
        Accuracy: 0.9583931133428981
        Precision: 0.8495145631067961
        Recall: 0.8663366336633663
        FPR: 0.026006711409395974
        FNR: 0.13366336633663367

        * feature_restriction = 20
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 81  | (FN) 121 |
        |    0     |  (FP) 0  |(TN) 1192 |
        Accuracy: 0.9131994261119082
        Precision: 1.0
        Recall: 0.400990099009901
        FPR: 0.0
        FNR: 0.599009900990099
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
        * feature_restriction = 0 (with all features)
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 181 | (FN) 21  |
        |    0     | (FP) 14  |(TN) 1178 |
        Accuracy: 0.9748923959827833
        Precision: 0.9282051282051282
        Recall: 0.8960396039603961
        FPR: 0.01174496644295302
        FNR: 0.10396039603960396

        * feature_restriction = 20
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 107 | (FN) 95  |
        |    0     |  (FP) 0  |(TN) 1192 |
        Accuracy: 0.9318507890961263
        Precision: 1.0
        Recall: 0.5297029702970297
        FPR: 0.0
        FNR: 0.47029702970297027
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

    def test_accuracy_baseline_wo_noise_w_bootstrap_with_10_features(self):
        """
        * feature_restriction = 0 (with all 10 features)
        |          |    1     |    0     |
        |----------|----------|----------|
        |    1     | (TP) 118 | (FN) 84  |
        |    0     | (FP) 15  |(TN) 1177 |
        Accuracy: 0.9289813486370158
        Precision: 0.8872180451127819
        Recall: 0.5841584158415841
        FPR: 0.012583892617449664
        FNR: 0.4158415841584158
        :return:
        """
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(numMutualInformationWords=5,
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
