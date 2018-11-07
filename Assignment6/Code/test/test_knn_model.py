import unittest

from Assignment5.Code import kDataPath
from model.KnnModel import KNearestNeighborModel
from utils.Assignment5Support import LoadRawData, Featurize, TrainTestSplit
from utils.EvaluationsStub import Evaluation


class TestKNearestNeighborModel(unittest.TestCase):

    def test_knn_w_ks(self):
        (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)
        (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw, includeGradients=True)
        yTrains = yTrainRaw
        yTests = yTestRaw
        for k in [100, 50, 20, 10, 1]:
            knn = KNearestNeighborModel(xTrains, yTrains)
            yTestPredictions = knn.predict(xTests, k)
            ev = Evaluation(yTests, yTestPredictions)
            print("K = ", k)
            print(ev)

    def test_knn_w_thresholds(self):
        (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)
        (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw, includeGradients=True)
        yTrains = yTrainRaw
        yTests = yTestRaw
        k = 10
        N = 10
        thresholds = [x / N for x in range(N + 1)]
        for threshold in thresholds:
            knn = KNearestNeighborModel(xTrains, yTrains)
            yTestPredictions = knn.predict(xTests, k, threshold)
            ev = Evaluation(yTests, yTestPredictions)
            print(ev)


if __name__ == '__main__':
    unittest.main()
