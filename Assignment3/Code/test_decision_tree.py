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

    def test_get_entropy(self):

        # all ys are 0s
        xTrains = [[1, 1]] * 10  # two features, ten samples
        yTrains = [0] * 10
        entropy = dtm.get_entropy(xTrains, yTrains)
        self.assertTrue(entropy == [0, 0], entropy)

        # all ys are 1s
        yTrains = [1, 1] * 10
        entropy = dtm.get_entropy(xTrains, yTrains)
        self.assertTrue(entropy == [0, 0], entropy)

        # 5:5
        yTrains = [1] * 5 + [0] * 5
        entropy = dtm.get_entropy(xTrains, yTrains)
        self.assertTrue(entropy == [1, 1], entropy)



if __name__ == '__main__':
    unittest.main()