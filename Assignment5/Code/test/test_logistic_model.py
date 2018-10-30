import os
import unittest

import model.LogisticRegressionModel as lrm
from Assignment4.Code import kDataPath
from assignments.previous.features_by_frequency import extract_features_by_frequency
from assignments.previous.features_by_mi import extract_features_by_mi
from utils.Assignment4Support import TrainTestSplit, LoadRawData, Featurize, FeaturizeTrainingByWords
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


class TestLogisticModel(unittest.TestCase):

    def test_predict_with_features_by_mi(self):

        # Loading data
        (xRaw, yRaw) = LoadRawData(kDataPath)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw)
        yTrain = yTrainRaw
        yTest = yTestRaw

        features, _ = extract_features_by_mi(xTrainRaw, yTrainRaw, N=10)
        features = [x[0] for x in features]

        model = lrm.LogisticRegressionModel(initial_weights=[.0] * len(features))
        xTrain = FeaturizeTrainingByWords(xTrainRaw, features)
        xTest = FeaturizeTrainingByWords(xTestRaw, features)

        # Extend xTrains and xTest with 1 at [0]
        xTrain = [[1] + x for x in xTrain]
        xTest = [[1] + x for x in xTest]

        model.fit(xTrain, yTrain, iterations=10000)
        yTestPrediced = model.predict(xTest)

        ev = Evaluation(yTest, yTestPrediced)
        print(ev)

    def test_predict_with_features(self):

        # Loading data
        (xRaw, yRaw) = LoadRawData(kDataPath)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw)
        yTrain = yTrainRaw
        yTest = yTestRaw

        features = extract_features_by_frequency(xTrainRaw, N=20)
        features = [x[0] for x in features]

        model = lrm.LogisticRegressionModel(initial_weights=[0.] * len(features))
        xTrain = FeaturizeTrainingByWords(xTrainRaw, features)
        xTest = FeaturizeTrainingByWords(xTestRaw, features)

        # Extend xTrains and xTest with 1 at [0]
        xTrain = [[1] + x for x in xTrain]
        xTest = [[1] + x for x in xTest]

        model.fit(xTrain, yTrain, iterations=10000)
        yTestPrediced = model.predict(xTest)

        ev = Evaluation(yTest, yTestPrediced)
        print(ev)

    def test_predict_baseline(self):

        # Loading data
        (xRaw, yRaw) = LoadRawData(kDataPath)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw)
        yTrain = yTrainRaw
        yTest = yTestRaw

        model = lrm.LogisticRegressionModel()
        #xTrain, xTest = Featurize(xTrainRaw, xTestRaw)
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(with_noise=False)
        # Extend xTrains and xTest with 1 at [0]
        xTrain = [[1] + x for x in xTrain]
        xTest = [[1] + x for x in xTest]
        model.weights = [0.] * (len(xTrain[0]))

        model.fit(xTrain, yTrain, iterations=10000)
        yTestPrediced = model.predict(xTest)

        ev = Evaluation(yTest, yTestPrediced)
        print(ev)

if __name__ == '__main__':
    unittest.main()
