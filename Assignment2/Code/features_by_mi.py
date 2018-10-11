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

# MI calculation

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
    raise ValueError('Missing keys() {} vs {}'.format(
        len(y0_counter.keys()), len(y1_counter.keys())))


def calculate_mi(y0_counter, y1_counter, top=10):

    features = y0_counter.keys()
    y0_N = sum(y0_counter.values())
    y1_N = sum(y1_counter.values())
    N = y0_N + y1_N
    mi = collections.Counter()

    for f in features:

        if not f:
            continue

        p_f_0 = (y0_counter[f] + 1) / (N + 2)
        p_0 = (y0_counter[f] + 1) / (y0_N + 2)
        total_cnt = y0_counter[f] + y1_counter[f]
        p_f = (y0_counter[f] + 1) / (total_cnt + 2)
        #mi_0 = p_f_0 * math.log2(p_f_0 / (p_f * p_0))
        mi_0 = p_f_0 * math.log2((N * y0_counter[f] + 1) / ((y0_N)*(total_cnt) * 2))

        p_f_1 = (y1_counter[f] + 1) / (N + 2)
        p_1 = (y1_counter[f] + 1) / (y1_N + 2)
        total_cnt = y0_counter[f] + y1_counter[f]
        p_f = (y1_counter[f] + 1) / (total_cnt + 2)
        #mi_1 = p_f_1 * math.log2(p_f_1 / (p_f * p_1))
        mi_1 = p_f_1 * math.log2((N * y1_counter[f] + 1) / ((y1_N)*(total_cnt) * 2))

        mi[f] = mi_0 + mi_1

    return mi.most_common(top)


def calculate_mi2(y0_counter, y1_counter, top=10):

    features = y0_counter.keys()
    y0_N = sum(y0_counter.values())
    y1_N = sum(y1_counter.values())
    N = y0_N + y1_N
    mi = collections.Counter()
    mi_tables = dict()
    for f in features:
        if not f:
            continue
        # For 'Call'

        #          |   'call'  | no 'Call' | ...
        # |--------|-----------|-----------|...
        # | y(=1)  |    n11    |    n10    |...
        # | y(=0)  |    n01    |    n00    |...
        n11 = y1_counter[f]         # number of 'Call' when y = 1
        n10 = y1_N - y1_counter[f]  # number of no 'Call' when y = 1
        n01 = y0_counter[f]         # number of 'Call' when y = 0
        n00 = y0_N - y0_counter[f]  # number of no 'Call' when y = 0
        
        n = n00 + n01 + n10 + n11
        n1_ = n10 + n11
        n_1 = n01 + n11
        n0_ = n00 + n01
        n_0 = n00 + n10
        
        mi[f] = (n11/n) * math.log2((n*n11 + 1) / (n1_ * n_1))\
                + (n01/n) * math.log2((n*n01 + 1) / (n0_ * n_1))\
                + (n10/n) * math.log2((n*n10 + 1) / (n1_ * n_0))\
                + (n00/n) * math.log2((n*n00 + 1) / (n0_ * n_0))
        mi_tables[f] = utils.table_for_mi(n11, n10, n01, n00, f)
    
    tops = mi.most_common(top)
    tables = [mi_tables[x[0]] for x in tops]
    return mi.most_common(top), tables

#features = calculate_mi(y0_counter, y1_counter, top=10)
features, mi_tables = calculate_mi2(y0_counter, y1_counter, top=10)
# print_lookup_table(y0_counter, y1_counter)

# Create a lookup table
#          |   'call'  |   'OK'   | ...
# |--------|-----------|----------|...
# | y(=0)  |     5     |    2     |...
# | y(=1)  |     3     |    1     |...

# note from Geoff:
# p(call, 0) would be 5 / 11 (= 5 / N)
# p(0) would be 5/7
# p(call) would be 5/8
# but then smooth them by adding 1 to the numerator and 2 to the denominator


table = utils.selected_features_table(
    features, ["Features", "Mutual Information"], w=25)

table_md = os.path.join(report_path, 'features_selected_by_mi.md')

with open(table_md, 'w') as f:
    f.write(table)
    f.write('\n')
    f.write('\nMutual Information Tables for Top {} features'.format(N))
    f.write('\n')
    for mi_table in mi_tables:
        f.write('\n')
        f.write(mi_table)
        f.write('\n')

