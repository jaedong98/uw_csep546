import collections
import itertools
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


def fold_data(xTrainRaw, k):
    """
    Args:
        xTrainRaw: a list of xTrainRaw data
        k: number of folding
    Returns:
        (a list of training sets, a list of validation sets)
    """
    if not k > 1:
        raise ValueError("Expected {} > 1".format(k))

    # divid xTrainRaw data into k group
    grouped_xTrainRaw = divide_into_group(xTrainRaw, k)
    print("Groupped xTrainRaw into {} groups.".format(k))

    trains = []
    validations = []
    for i in range(k):
        group = list(grouped_xTrainRaw)
        validation = group.pop(i)
        train = list(itertools.chain.from_iterable(group))
        trains.append(train)
        validations.append(validation)

    return trains, validations


def divide_into_group(xTrainRaw, k):

    cnt = len(xTrainRaw) / k
    groups = []
    last = 0.0

    while last < len(xTrainRaw):
        groups.append(xTrainRaw[int(last):int(last + cnt)])
        last += cnt

    if not len(xTrainRaw) == sum([len(g) for g in groups]):
        raise AssertionError("Missing/Duplicated element {} vs {}"
                             .format(len(xTrainRaw),
                                     sum([len(g) for g in groups])))

    if not len(groups) == k:
        raise AssertionError("More or less groups found {} vs (expected){}"
                             .format(len(groups), k))
    return groups


def get_features_by_frequency(xTrainRaw, xTestRaw N):

    features = extract_features_by_frequency(xTrainRaw, N)
    features = [x[0] for x in features]
    xTrain = FeaturizeTrainingByWords(xTrainRaw, features)
    xTest = FeaturizeTrainingByWords(xTestRaw, features)

    return xTrain, xTest


def run_gradient_descent(xTrain, xTest, yTrain, yTest, N=10,
                         max_iters=50000, iter_step=1000, step=0.01,
                         initial_w0=0.0, report_path=report_path, fname='', k=5):
    """
    Returns: iter_cnt_vs_loss, iter_cnt_vs_accuracy
    """

    fold_xTrains, fold_xVals = fold_data(xTrain, k)
    fold_yTrains, fold_yVals = fold_data(yTrain, k)

    i = 0
    total_correct = 0
    for f_xTrain, f_xVal, f_yTrain, f_yVal in zip(fold_xTrains, fold_xVals,
                                                  fold_yTrains, fold_yVals):

        print("Gradient descent for {}th folding".format(i))
        resolution = int(max_iters / iter_step)
        features = [x[0] for x in features]

        model = utils.logistic_regression_model_by_features(
            f_xTrain, f_yTrain, features, iter_step, resolution, initial_w0, step, max_iters)
        f_yVal_predict = model.predict(f_xVal)

        for p, v in zip(f_yVal_predict, f_yVal):
            if p == v:
                total_correct += 1

        print("Total correction: {}".format(total_correct))

        i += 1

    accuracy = total_correct / len(xTrain)
    print("Accuracy: {}".format(accuracy))

    return accuracy


def compare_models(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                   N=10,
                   max_iters=50000,
                   iter_step=1000,
                   step=0.01,
                   initial_w0=0.0,
                   k=5,
                   zn=1.96,
                   report_path=report_path):

    yTrain = yTrainRaw
    yTest = yTestRaw

    # Top N frequency
    xTrain, xTest = get_features_by_frequency(xTrainRaw, xTestRaw, N)
    accuracy_by_frequency = run_gradient_descent(xTrain, xTest, yTrain, yTest,
                                                 N=N,
                                                 max_iters=max_iters,
                                                 iter_step=iter_step,
                                                 step=step,
                                                 initial_w0=initial_w0,
                                                 k=k)
    #ub_f, lb_f = utils.calculate_bounds(accuracy_by_frequency, zn, len(xTrain))

    # Top N Mutual Information
    accuracy_by_mi = run_gradient_descent(xTrain, xTest, yTrain, yTest,
                                          N=N,
                                          max_iters=max_iters,
                                          iter_step=iter_step,
                                          step=step,
                                          initial_w0=initial_w0,
                                          k=k)
    #ub_m, lb_m = utils.calculate_bounds(accuracy_by_frequency, zn, len(xTrain))

    # report table (including upper and lower bounds)
    table = utils.table_for_gradient_accuracy_estimate(accuracies, legends,
                                                       len(xTrain), zn)
    print(table)
    cm_md = os.path.join(report_path,
                         '{}_N{}_k{}.md'.format(inspect.stack()[0][3], N, k))
    with open(cm_md, 'w') as f:
        f.write(table)


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = utils.TrainTestSplit(xRaw,
                                                                      yRaw)

    N = 10
    max_iters = 50000
    iter_step = 1000
    step = 0.01
    initial_w0 = 0.0
    k = 5
    zn = 1.96
    compare_models(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                   N=N,
                   max_iters=max_iters,
                   iter_step=iter_step,
                   step=step,
                   initial_w0=initial_w0,
                   k=k,
                   zn=zn)
