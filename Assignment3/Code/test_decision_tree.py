import unittest
import os
import Assignment3Support as utils
import DecisionTreeModel as dtm

# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


# Loading data
(xRaw, yRaw) = utils.LoadRawData(kDataPath)

# Train-Test split
# TODO: splitting data into train, validation, test?
(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrains, xTests) = utils.Featurize(xTrainRaw, xTestRaw)
yTrains = yTrainRaw
yTests = yTestRaw


class TestDecisionTreeModel(unittest.TestCase):

    def test_get_split(self):
        pass

    def test_entropy_S(self):

        # Mitchell, page 56, when all values are 0
        yTrains = [0] * 10
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 0)

        # Mitchell, page 56, when all values are 1
        yTrains = [1] * 10
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 0)

        # Mitchell, page 56, when num(0s) == num(1s)
        yTrains = [1] * 5 + [0] * 5
        s = dtm.get_entropy_S(yTrains)
        self.assertTrue(s == 1)

        # case - Mitchell, Chapter 3, page 56
        yTrains = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        s = dtm.get_entropy_S(yTrains)
        self.assertAlmostEqual(s, 0.940, 3)

    def test_get_entropy_for_feature(self):

        # Mitchell, page 58
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        feature_dict = dtm.get_feature_dict(wind, play_tennis)
        es = dtm.get_entropy_for_feature(feature_dict)
        for entropy, expected in zip(es, [0.811, 1.0]):
            self.assertAlmostEqual(entropy, expected, 3)

    def test_get_information_gain(self):

        # Mitchell, page 58
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        gain = dtm.get_information_gain(wind, play_tennis)
        self.assertAlmostEqual(gain, 0.048, 3)

    def test_get_information_gains(self):
        """Construct xTrains data to have same data structure from original framework."""
        # Mitchell, page 59 and 60
        humidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        gains = dtm.get_information_gains(xTrains, play_tennis)
        for gain, expected in zip(gains, [0.151, 0.048]):
            self.assertAlmostEqual(gain, expected, 2)


if __name__ == '__main__':
    unittest.main()