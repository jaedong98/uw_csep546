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


def run_gradient_descent(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, N=10,
                         max_iters=50000, iter_step=1000, step=0.01,
                         initial_w0=0.0):

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
            
    features, mi_tables = calculate_mi2(y0_counter, y1_counter, top=N)

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

    # gradient decent
    tic = time.time()
    iter_cnts = [0]
    resolution = int(max_iters / iter_step)

    features = [x[0] for x in features]
    xTrain = utils.FeaturizeTrainingByWords(xTrainRaw, features)
    xTest = utils.FeaturizeTrainingByWords(xTestRaw, features)

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

    iter_cnt_vs_loss_png = os.path.join(
        report_path, 'iter_cnt_vs_accuracy_by_mi_{}.png'.format(max_iters))
    title = 'Loss with top 10 frequent words'.format(max_iters)
    utils.draw_accuracies([iter_cnt_vs_accuracy], 'Iterations', 'Accuracy',
                          title, iter_cnt_vs_loss_png, ['Accuracy by MI'])

    return iter_cnt_vs_loss, iter_cnt_vs_accuracy


if __name__ == '__main__':
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    run_gradient_descent(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,  N=10,
                         max_iters=500, iter_step=100, step=0.01,
                         initial_w0=0.0)