import collections
import matplotlib.pyplot as plt
import math
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

feature_selection_methods = [utils.is_longger,
                             utils.has_number,
                             utils.contain_call,
                             utils.contain_to,
                             utils.contain_your]

xTrain = utils.FeaturizeTraining(xTrainRaw, feature_selection_methods)


## MI calculation

y0_counter = collections.Counter()
y1_counter = collections.Counter()
for y, x in zip(yTrainRaw, xTrainRaw):
    for w in x.split(' '):
        if y == 0:
            y0_counter[w] += 1
            y1_counter[w] += 0
        else:
            y0_counter[w] += 0
            y1_counter[w] += 1

if not len(y0_counter.keys()) == len(y1_counter.keys()):
    raise ValueError('Missing keys() {} vs {}'.format(len(y0_counter.keys()),len(y1_counter.keys())))


def calculate_mi(y0_counter, y1_counter, top=10):

    features = y0_counter.keys()
    y0_N = sum(y0_counter.values())
    y1_N = sum(y1_counter.values()) 
    N = y0_N + y1_N

    mi = collections.Counter()


    for f in features:

        p_f_0 = (y0_counter[f] + 1) / (N + 2)
        p_0 = (y0_counter[f] + 1) / (y0_N + 2)
        total_cnt = y0_counter[f] + y1_counter[f]
        p_f = (y0_counter[f] + 1) / (total_cnt + 2)
        mi_0 = p_f_0 * math.log2(p_f_0 / (p_f * p_0))

        p_f_1 = (y1_counter[f] + 1) / (N + 2)
        p_1 = (y1_counter[f] + 1) / (y1_N + 2)
        total_cnt = y0_counter[f] + y1_counter[f]
        p_f = (y0_counter[f] + 1) / (total_cnt + 2)
        mi_1 = p_f_1 * math.log2(p_f_1 / (p_f * p_1))


        mi[f] = mi_0 + mi_1

    return mi.most_common(top)

features = calculate_mi(y0_counter, y1_counter, top=10)
# print_lookup_table(y0_counter, y1_counter)

# Create a lookup table
#         |   'call'  |   'OK'   | ...
#|--------|-----------|----------|...
#| y(=0)  |     5     |    2     |...
#| y(=1)  |     3     |    1     |...

# note from Geoff:
# p(call, 0) would be 5 / 11 (= 5 / N)
# p(0) would be 5/7
# p(call) would be 5/8
# but then smooth them by adding 1 to the numerator and 2 to the denominator

table = utils.selected_features_table(features, ["Features", "Mutual Information"], w=25)

table_md = os.path.join(report_path, 'features_selected_by_mi.md')

with open(table_md, 'w') as f:
    f.write(table)