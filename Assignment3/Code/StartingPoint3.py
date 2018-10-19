import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import Assignment3Support as utils
import EvaluationsStub
import LogisticRegressionModel as lgm
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

(xTrain, xTest) = utils.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

min_to_stop = 100
accuracy_md = os.path.join(report_path, 'prob2_part1_accuracy.md')
model = dtm.DecisionTreeModel()
model.fit(xTrain, yTrain, min_to_stop)
model.visualize()
with open(accuracy_md, 'w') as file_obj:
    file_obj.write('Decision Tree with minToStop={}'.format(min_to_stop))
    model.visualize(file_obj)

