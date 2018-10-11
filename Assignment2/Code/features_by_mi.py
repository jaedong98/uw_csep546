import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import Assignment2Support as utils
import EvaluationsStub
import LogisticRegressionModel as lgm

# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")

N = 10

# Loading data
(xRaw, yRaw) = utils.LoadRawData(kDataPath)

# Train-Test split
# TODO: splitting data into train, validation, test?
(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)

## MI calculation

count = collections.Counter()
for x in xTrainRaw:
    for w in x.split(' '):
        count[w] += 1

features = count.most_common(N)
for y, row in zip(yTrainRaw, xTrainRaw):
    
    mi = 0
    for y in count.keys():
        for x in count.keys():
            if feature == feature:
                continue
            

features = []

table = utils.selected_features_table(features, ["Features", "Frequency"])

table_md = os.path.join(report_path, 'features_selected_by_mi_table_{}.md'.format(N))

with open(table_md, 'w') as f:
    f.write(table)