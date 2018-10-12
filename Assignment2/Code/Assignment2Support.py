import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import LogisticRegressionModel as lgm
import EvaluationsStub


def LoadRawData(path):
    f = open(path, 'r')

    lines = f.readlines()

    kNumberExamplesExpected = 5574

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (
            kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)


def TrainTestSplit(x, y, percentTest=.25):
    if(len(x) != len(y)):
        raise UserWarning(
            "Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning(
            "Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)


def GetAllDataExceptFold(xRaw, yRaw, i):
    pass


def GetDataInFold(xRaw, yRaw, i):
    pass


def FeaturizeTraining(xTrainRaw, feature_selection_functions):
    """
    xTrainRaw: a list of spam messages.
    feature_section_functions: a list of f(msg)s
    """
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        for fsf in feature_selection_functions:
            features.append(fsf(x))

        xTrain.append(features)

    return xTrain


def FeaturizeTrainingByWords(xTrainRaw, words):
    """
    xTrainRaw: a list of spam messages.
    features: a list of features(words)
    """
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        for f in words:
            if f in x:
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    return xTrain


def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])


def most_frequent_features(xTrain, N=1):

    count = collections.Counter()
    for x in xTrain:
        words = x.split(' ')
        for word in words:
            count[word] += 1

    return count.most_common(N)


def is_longger(msg, threshold=40):
    if len(msg) > threshold:
        return 1
    return 0


def has_number(msg):
    if any(i.isdigit() for i in msg):
        return 1
    return 0


def contain_word(msg, word):
    if word in msg:
        return 1
    return 0


def contain_call(msg):
    return contain_word(msg, 'call')


def contain_to(msg):
    return contain_word(msg, 'to')


def contain_your(msg):
    return contain_word(msg, 'your')


def exclude(msg):
    return 1


def draw_single_plot(tuples, xlabel, ylabel, title, img_fname):
    t, s = zip(*tuples)
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def draw_weights(iter_cnts, weights, xlabel, ylabel, title, img_fname):
    """
    iter_cnts: a list of iteration counts, [1000, 2000, ...]
    weights: a list of weights per iteration, [(0, 0,...), (0.1, 0.2,...),...]
    """
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for ws in zip(*weights):
        ax.plot(iter_cnts, ws)

    ax.legend(('w0', 'w1', 'w2', 'w3', 'w4', 'w5'), loc='best')
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def draw_accuracies(accuracies, xlabel, ylabel, title, img_fname, legends):
    """
    accuracies: a list of list of (iter_cnt, accurarcy)s
    legends: a tuple/list of legends for each graphs
    """
    if legends:
        if not len(accuracies) == len(legends):
            raise ValueError("Missing legends? {} vs {}".format(
                len(accuracies), len(legends)))

    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    for accus in accuracies:
        xs, ys = zip(*accus)
        ax.plot(xs, ys)

    if legends:
        ax.legend(legends, loc='best')
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def accuracy_table(accuracies, features, w=20):

    table = '|{}|{}|'.format(
        'Leave-out-Features'.center(w), 'Accuracy'.center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|'
    for feature, accu in zip(features, accuracies):
        table += '\n|{}|{}|'.format('{}'.format(feature).center(w),
                                    '{}'.format(accu).center(w))
    return table


def selected_features_table(features, headers, w=20):
    """
    features: a list of two element tuples, [('to', 1586), ('you', 1174),...]
    headers: a list of two headers, ["Feature", "Frequency"]
    """
    table = '|{}|{}|'.format(headers[0].center(w), headers[1].center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|'
    for feature, num in features:
        table += '\n|{}|{}|'.format('{}'.format(feature).center(w),
                                    '{}'.format(num).center(w))
    return table


def table_for_mi(n11, n10, n01, n00, feature, w=15):
    """
    Creates a string of table.
    n11, n10, n01, n00: integer
    feature: a string

    Returns:
        a string of following table.
        |        |   'call'  | no 'Call' |
        |--------|-----------|-----------|
        | y(=1)  |    n11    |    n10    |
        | y(=0)  |    n01    |    n00    |
    """

    table = '|{}|{}|{}|'.format(''.center(w), feature.center(
        w), 'No "{}"'.format(feature).center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|' + '-' * w
    table += '|'

    table += '\n|{}|{}|{}|'.format('y(=1)'.center(w),
                                   '{}'.format(n11).center(w), '{}'.format(n10).center(w))
    table += '\n|{}|{}|{}|'.format('y(=0)'.center(w),
                                   '{}'.format(n01).center(w), '{}'.format(n00).center(w))

    return table


def logistic_regression_by_features(xTrainRaw, xTestRaw, yTrain, yTest, features, iter_step, resolution, initial_w0, step, max_iters, img_fname=None, title=""):
    """
    Args:
        features: a list of features(words)
    """
    xTrain = FeaturizeTrainingByWords(xTrainRaw, features)
    xTest = FeaturizeTrainingByWords(xTestRaw, features)

    model = lgm.LogisticRegressionModel(initial_w0=initial_w0,
                                        initial_weights=[0.0] * len(features))

    # Extend xTrains and xTest with 1 at [0]
    xTrain = [[1] + x for x in xTrain]
    xTest = [[1] + x for x in xTest]

    iter_cnt_vs_loss = []
    iter_cnt_vs_accuracy = []
    for i, iters in enumerate([iter_step] * resolution):
        fit_tic = time.time()
        model.fit(xTrain, yTrain, iterations=iters, step=step)
        fit_toc = time.time() - fit_tic
        iter_cnt = iter_step * (i + 1)
        print("Took {} sec. Fitted data for {} iterations".format(fit_toc, iter_cnt))
        yTestPredicted = model.predict(xTest)
        test_loss = model.loss(xTest, yTest)
        iter_cnt_vs_loss.append((iter_cnt, test_loss))
        test_accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
        iter_cnt_vs_accuracy.append((iter_cnt, test_accuracy))
        print("%d, %f, %f" % (iter_cnt, test_loss, test_accuracy))

    if img_fname:
        draw_accuracies([iter_cnt_vs_accuracy], 'Iterations', 'Accuracy',
                        title, img_fname, [])

    return iter_cnt_vs_loss, iter_cnt_vs_accuracy


def logistic_regression_model_by_features(xTrainRaw, yTrain, features, iter_step, resolution, initial_w0, step, max_iters):
    """
    Featuring xTrainRaw data and run gradient descents.
    Returns:
        an instance of Logistic Regression model
    """
    xTrain = FeaturizeTrainingByWords(xTrainRaw, features)

    model = lgm.LogisticRegressionModel(initial_w0=initial_w0,
                                        initial_weights=[0.0] * len(features))

    # Extend xTrains and xTest with 1 at [0]
    xTrain = [[1] + x for x in xTrain]

    for i, iters in enumerate([iter_step] * resolution):
        fit_tic = time.time()
        model.fit(xTrain, yTrain, iterations=iters, step=step)
        fit_toc = time.time() - fit_tic
        iter_cnt = iter_step * (i + 1)
        print("Took {} sec. Fitted data for {} iterations".format(fit_toc, iter_cnt))

    return model


def table_for_gradient_accuracy_comparision(accuracies, legends, w=30):
    """
    Args:
        accuracies: a list of accuracy lists from different gradient decents.
        legends: a list of strings to be named. ['Top 10 frequent words', ...]
    """
    table = '|{}|{}|'.format(
        'Configurations'.center(w), 'Accuracy'.center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|'
    for legend, accu in zip(legends, accuracies):
        table += '\n|{}|{}|'.format('{}'.format(legend).center(w),
                                    '{}'.format(accu[-1][-1]).center(w))
    return table


def table_for_gradient_accuracy_estimate(accuracies, legends, N, zn=1.96, w=30):
    """
    Args:
        accuracies: a list of accuracy lists from different gradient decents.
        legends: a list of strings to be named. ['Top 10 frequent words', ...]
        N: len(xTrains)
    """
    table = '|{}|{}|{}|{}|'.format('Feature Selections'.center(w),
                                   'Accuracy'.center(w),
                                   'Upper'.center(w),
                                   'Lower'.center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|' + '-' * w
    table += '|' + '-' * w
    table += '|'

    for legend, accu in zip(legends, accuracies):
        accuracy = accu[-1][-1]
        upper, lower = calculate_bounds(accuracy, zn, N)
        table += '\n|{}|{}|{}|{}|'.format('{}'.format(legend).center(w),
                                          '{}'.format(accuracy).center(w),
                                          '{}'.format(upper).center(w),
                                          '{}'.format(lower).center(w))
    return table


def calculate_bounds(accuracy, zn, N):
    upper = accuracy + zn * np.sqrt((accuracy * (1 - accuracy) / N))
    lower = accuracy - zn * np.sqrt((accuracy * (1 - accuracy) / N))
    return upper, lower

#legends = ['apple', 'banana']
#accuracies = [[(0,0.9), (100, 0.8)],[(10, 0.7), (100, 0.91)]]
#print(table_for_gradient_accuracy_estimate(accuracies, legends, N=10))
