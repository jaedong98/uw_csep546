import collections
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import Assignment2Support as utils
import EvaluationsStub
import LogisticRegressionModel as lgm
import features_by_frequency as fbf
import features_by_mi as fbm

# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def accuracy_estimate_bounds(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, N, max_iters, iter_step, step, initial_w0, zn=1.96, features=[], report_path=report_path):
    losses = []
    accuracies = []
    legends = []
    # Top N frequency
    fname = '{}_by_frequency_{}.png'.format(inspect.stack()[0][3], max_iters)
    iter_cnt_vs_loss, iter_cnt_vs_accuracy, features_by_frequency = fbf.run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                                                                                             max_iters=max_iters, iter_step=iter_step, step=step,
                                                                                             initial_w0=initial_w0, fname=fname)
    losses.append(iter_cnt_vs_loss)
    accuracies.append(iter_cnt_vs_accuracy)
    legends.append('Top {} Frequency'.format(N))

    # Top N MI
    fname = '{}_by_mi_{}.png'.format(inspect.stack()[0][3], max_iters)
    title = "Accuracy Over Iteration by Top {} Frequency Features.".format(N)
    iter_cnt_vs_loss, iter_cnt_vs_accuracy, features_by_mi = fbm.run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                                                                                      max_iters=max_iters, iter_step=iter_step, step=step,
                                                                                      initial_w0=initial_w0, fname=fname)
    losses.append(iter_cnt_vs_loss)
    accuracies.append(iter_cnt_vs_accuracy)
    legends.append('Top {} MI'.format(N))

    # Outputs
    cnt = len(xTrainRaw)
    table = utils.table_for_gradient_accuracy_estimate(accuracies, legends, cnt, zn=zn)
    table_md = os.path.join(
        report_path, 'accuracy_estimates_train_test_split_{}.md'.format(N))
    print(table)
    with open(table_md, 'w') as f:
        f.write('* Accuracy Estimates w/ Zn={}'.format(zn))
        f.write('\n')
        f.write(table)


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    # Configuration
    max_iters = 50000
    iter_step = 1000
    step = 0.01
    initial_w0 = 0.0

    # Top 10
    N = 10
    accuracy_estimate_bounds(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, N, max_iters, iter_step, step, initial_w0)
