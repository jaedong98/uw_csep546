import collections
import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import Assignment2Support as utils
import EvaluationsStub as es
import LogisticRegressionModel as lgm
import features_by_frequency as fbf
import features_by_mi as fbm


# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


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


def find_categrize_mistakes(xTrain, xTest, yTrain, yTest, features, iter_step, resolution, intial_w0, step, max_iters, top=20):

    model = utils.logistic_regression_model_by_features(
        xTrain, yTrain, features, iter_step, resolution, initial_w0, step, max_iters)

    xTest = [[1] + x for x in xTest]
    # predict using validation dataset
    yTestPredicted_prob = model.predict_probabilities(xTest)
    yTestPredicted = model.predict(xTest)

    fn = []
    fp = []

    for i, (t, p) in enumerate(zip(yTest, yTestPredicted)):
        prob = yTestPredicted_prob[i]
        if (t, p) == (1, 0):  # false negative
            fn.append((prob, i))
        elif (t, p) == (0, 1):  # false positive
            fp.append((prob, i))

    # the true answer was 1, but the model gives very low probabilities
    sorted_fn = sorted(fn)
    # the true answer was 0, but gives very high probabilities.
    sorted_fp = sorted(fp, reverse=True)
    print('*' * 80)
    print('Total {} false netagives.'.format(len(sorted_fn)))
    print('Total {} false positives.'.format(len(sorted_fp)))
    print('False Netagives: {}'.format(sorted_fn))
    print('False Positives: {}'.format(sorted_fp))
    print(es.ConfusionMatrix(yTest, yTestPredicted))
    print('*' * 80)
    return sorted_fn[:top], sorted_fp[:top]


def generate_mistakes_table(mistakes, title, header, xTestRaw, fname, w=30):

    top_mistakes = title
    top_mistakes += '\n'
    top_mistakes += '\n  {}'.format(header)
    top_mistakes += '\n  |-|-|'
    for prob, i in mistakes:
        top_mistakes += '\n  |{}| {}|'.format(
            '{}'.format(prob).center(w), xTestRaw[i].strip())

    with open(fname, 'w') as f:
        f.write(top_mistakes)
    print("Created {}".format(fname))


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = utils.TrainTestSplit(xRaw,
                                                                      yRaw)

    N = 10
    max_iters = 50000
    iter_step = 1000
    resolution = int(max_iters / iter_step)
    initial_w0 = 0.0
    step = 0.01
    top = 20

    ############################################################################
    # by frequency
    xTrain, xTest, features = get_features_by_frequency(xTrainRaw, xTestRaw, N)
    yTrain = yTrainRaw
    yTest = yTestRaw
    _, sorted_fp = find_categrize_mistakes(
        xTrain, xTest, yTrain, yTest, features, iter_step, resolution, initial_w0, step, top)

    w = 30
    header = '| Probabilities | Test Raw |'
    title = '* False Positive - the true answer was 0, but gives very high probabilities'
    fname = os.path.join(report_path, 'category_mistake_false_positives.md')
    generate_mistakes_table(sorted_fp, title, header, xTestRaw, fname)

    ############################################################################
    # by mutual information
    xTrain, xTest, features = get_features_mi(
        xTrainRaw, yTrainRaw, xTestRaw, N)
    yTrain = yTrainRaw
    yTest = yTestRaw
    sorted_fn, _ = find_categrize_mistakes(
        xTrain, xTest, yTrain, yTest, features, iter_step, resolution, initial_w0, step, top)

    w = 30
    title = '* False Negatives - the true answer was 1, but the model gives very low probabilities'
    fname = os.path.join(report_path, 'category_mistake_false_negatives.md')
    generate_mistakes_table(sorted_fn, title, header, xTestRaw, fname)
    ############################################################################
