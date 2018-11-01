import unittest

from Assignment5.Code import kDataPath
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize
from utils.k_means_clustring import KMeanClustring


class TestKMeanClustering(unittest.TestCase):

    def test_k_means_in_x_direction(self):

        xTrains = [[0, 0], [4, 0], [8, 0], [16, 0]]
        k = 1
        iterations = 10
        kmc = KMeanClustring(xTrains, k, iterations)
        kmc.cluster()
        self.assertTrue(len(kmc.centroids) == 1)
        self.assertAlmostEqual(kmc.centroids[0].x, 7.0, 8)
        self.assertAlmostEqual(kmc.centroids[0].y, 0.0, 8)
        print(kmc.closest_pairs())

    def test_k_means_in_y_direction(self):

        xTrains = [[0, 0], [0, 4], [0, 8], [0, 16]]
        k = 1
        iterations = 10
        kmc = KMeanClustring(xTrains, k, iterations)
        kmc.cluster()
        self.assertTrue(len(kmc.centroids) == 1)
        self.assertAlmostEqual(kmc.centroids[0].x, 0.0, 8)
        self.assertAlmostEqual(kmc.centroids[0].y, 7.0, 8)

    def test_k_means_on_training_data(self):
        (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
        (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)
        (xTrains, _) = Featurize(xTrainRaw, xTestRaw, includeGradients=True)
        k = 2
        iterations = 2
        kmc = KMeanClustring(xTrains, k, iterations)
        kmc.cluster()
        self.assertTrue(len(kmc.centroids) == 2)

        print(kmc.closest_pairs())


if __name__ == '__main__':
    unittest.main()
