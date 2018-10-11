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


def get_most_frequent_features(xTrainRaw, N):

    count = collections.Counter()
    for x in xTrainRaw:
        for w in x.split(' '):
            count[w] += 1

    return count.most_common(N)


def run_gradient_descent(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, N=10,
                         max_iters=50000, iter_step=1000, step=0.01,
                         initial_w0=0.0):

    features = get_most_frequent_features(xTrainRaw, N)

    table = utils.selected_features_table(features, ["Features", "Frequency"])

    table_md = os.path.join(
        report_path, 'features_selected_by_top_{}_frequenct_words.md'.format(N))

    with open(table_md, 'w') as f:
        f.write(table)

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
        report_path, 'iter_cnt_vs_accuracy_by_frequency_{}.png'.format(max_iters))
    title = 'Loss with top 10 frequent words'.format(max_iters)
    utils.draw_accuracies([iter_cnt_vs_accuracy], 'Iterations', 'Accuracy',
                          title, iter_cnt_vs_loss_png, ['Accuracy by Word Frequency'])

    return iter_cnt_vs_loss, iter_cnt_vs_accuracy


if __name__ == '__main__':
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    run_gradient_descent(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,  N=10,
                         max_iters=500, iter_step=100, step=0.01,
                         initial_w0=0.0)
