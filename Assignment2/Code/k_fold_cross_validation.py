import collections
import inspect
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


def get_features_by_frequency(xTrainRaw, xTestRaw, N):
    """
    Returns: a tuple of (xTrain, xTest, a list of words(features))
    """
    features = fbf.extract_features_by_frequency(xTrainRaw, N)
    features = [x[0] for x in features]
    xTrain = utils.FeaturizeTrainingByWords(xTrainRaw, features)
    xTest = utils.FeaturizeTrainingByWords(xTestRaw, features)

    return xTrain, xTest, features


def get_features_mi(xTrainRaw, yTrainRaw, xTestRaw, N):
    """
    Returns: a tuple of (xTrain, xTest, a list of words(features))
    """
    features, _ = fbm.extract_features_by_mi(xTrainRaw, yTrainRaw, N)
    features = [x[0] for x in features]
    xTrain = utils.FeaturizeTrainingByWords(xTrainRaw, features)
    xTest = utils.FeaturizeTrainingByWords(xTestRaw, features)

    return xTrain, xTest, features


def calculate_accuracy_by_cv(xTrain, xTest, yTrain, yTest, features, N=10,
                             max_iters=50000, iter_step=1000, step=0.01,
                             initial_w0=0.0, report_path=report_path, fname='',
                             k=5):
    """
    Calculatge the accracy of cross validation using gradient descent.
    Returns: accuracy from cross validation.
    """

    # folding data into Train vs Validation
    fold_xTrains, fold_xVals = fold_data(xTrain, k)
    fold_yTrains, fold_yVals = fold_data(yTrain, k)

    resolution = int(max_iters / iter_step)
    i = 0
    total_correct = 0
    for f_xTrain, f_xVal, f_yTrain, f_yVal in zip(fold_xTrains, fold_xVals,
                                                  fold_yTrains, fold_yVals):

        print("Gradient descent for {}th folding".format(i))
        model = utils.logistic_regression_model_by_features(
            f_xTrain, f_yTrain, features, iter_step, resolution, initial_w0, step, max_iters)

        # predict using validation dataset
        f_xVal = [[1] + x for x in f_xVal]
        f_yVal_predict = model.predict(f_xVal)

        # compare and count corrections
        for p, v in zip(f_yVal_predict, f_yVal):
            if p == v:
                total_correct += 1

        print("Total correction: {}".format(total_correct))

        i += 1

    accuracy = total_correct / len(xTrain)
    print("Accuracy: {}".format(accuracy))

    return accuracy


def compare_models_by_cross_validation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
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
    xTrain, xTest, f_features = get_features_by_frequency(
        xTrainRaw, xTestRaw, N)
    accuracy_by_frequency = calculate_accuracy_by_cv(xTrain, xTest, yTrain, yTest,
                                                     f_features,
                                                     N=N,
                                                     max_iters=max_iters,
                                                     iter_step=iter_step,
                                                     step=step,
                                                     initial_w0=initial_w0,
                                                     k=k)
    #ub_f, lb_f = utils.calculate_bounds(accuracy_by_frequency, zn, len(xTrain))

    # Top N Mutual Information
    xTrain, xTest, m_features = get_features_mi(
        xTrainRaw, yTrainRaw, xTestRaw, N)
    accuracy_by_mi = calculate_accuracy_by_cv(xTrain, xTest, yTrain, yTest,
                                              m_features,
                                              N=N,
                                              max_iters=max_iters,
                                              iter_step=iter_step,
                                              step=step,
                                              initial_w0=initial_w0,
                                              k=k)
    #ub_m, lb_m = utils.calculate_bounds(accuracy_by_frequency, zn, len(xTrain))

    # report table (including upper and lower bounds)
    accuracies = [accuracy_by_frequency, accuracy_by_mi]
    legends = ["Top 10 Frequency", "Top 10 MI "]
    table = utils.table_for_cross_validation_accuracy_estimate(accuracies,
                                                               legends,
                                                               len(xTrain),
                                                               N, zn)
    print(table)
    cm_md = os.path.join(report_path,
                         '{}_N{}_k{}.md'.format(inspect.stack()[0][3], N, k))
    with open(cm_md, 'w') as f:
        f.write(table)
        f.write("\n")
        f.write("\n Features by Frequency: {}".format(f_features))
        f.write("\n")
        f.write("\n Features by Mutual Information: {}".format(m_features))


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
    compare_models_by_cross_validation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                       N=N,
                                       max_iters=max_iters,
                                       iter_step=iter_step,
                                       step=step,
                                       initial_w0=initial_w0,
                                       k=k,
                                       zn=zn)
