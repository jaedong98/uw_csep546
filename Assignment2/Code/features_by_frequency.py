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


# Loading data
(xRaw, yRaw) = utils.LoadRawData(kDataPath)


def extract_features_by_frequency(xTrainRaw, N):

    count = collections.Counter()
    for x in xTrainRaw:
        for w in x.split(' '):
            count[w] += 1

    return count.most_common(N)


def run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                         max_iters=50000, iter_step=1000, step=0.01,
                         initial_w0=0.0):
    """
    Returns: iter_cnt_vs_loss, iter_cnt_vs_accuracy
    """
    features = extract_features_by_frequency(xTrainRaw, N)

    table = utils.selected_features_table(features, ["Features", "Frequency"])

    table_md = os.path.join(
        report_path, 'features_selected_by_top_{}_frequenct_words.md'.format(N))

    with open(table_md, 'w') as f:
        f.write(table)

    # gradient decent
    iter_cnts = [0]
    resolution = int(max_iters / iter_step)
    features = [x[0] for x in features]
    return utils.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                 yTrain, yTest,
                                                 features, iter_step,
                                                 resolution, initial_w0,
                                                 step, max_iters, report_path)


if __name__ == '__main__':
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                         max_iters=500, iter_step=100, step=0.01,
                         initial_w0=0.0)
