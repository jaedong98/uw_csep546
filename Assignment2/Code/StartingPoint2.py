import matplotlib
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

# Train-Test split
# TODO: splitting data into train, validation, test?
(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

#(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw


# TODO:
# functions to filter features
feature_selection_methods_options = [utils.is_longger,
                                     utils.has_number,
                                     utils.contain_call,
                                     utils.contain_to,
                                     utils.contain_your]

tic = time.time()
max_iters = 50000
iter_cnts = [0]
iter_step = 1000
resolution = int(max_iters / iter_step)

iter_cnt_vs_accuracies = []
legends = []
for i in range(len(feature_selection_methods_options)):
    feature_selection_methods = list(feature_selection_methods_options)
    function_out = feature_selection_methods.pop(i)
    legends.append("w/o {}".format(function_out.__name__.upper()))
    print("Exclude {}".format(function_out.__name__))
    xTrain = utils.FeaturizeTraining(xTrainRaw, feature_selection_methods)
    xTest = utils.FeaturizeTraining(xTestRaw, feature_selection_methods)
    model = lgm.LogisticRegressionModel(initial_w0=0.0,
                                        initial_weights=[0.0] * len(feature_selection_methods))
    weights = [list(model.weights)]
    # Extend xTrains and xTest with 1 at [0]
    xTrain = [[1] + x for x in xTrain]
    xTest = [[1] + x for x in xTest]

    iter_cnt_vs_accu = []
    for i, iters in enumerate([iter_step] * resolution):
        fit_tic = time.time()
        model.fit(xTrain, yTrain, iterations=iters, step=0.01)
        fit_toc = time.time() - fit_tic
        iter_cnt = iter_step * (i + 1)
        print("Took {} sec. Fitted data for {} iterations".format(fit_toc, iter_cnt))
        yTestPredicted = model.predict(xTest)
        test_loss = model.loss(xTest, yTest)
        test_accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
        print("%d, %f, %f, %f" %
              (iter_cnt, model.weights[1], test_loss, test_accuracy))
        iter_cnt_vs_accu.append((iter_cnt, test_accuracy))

    iter_cnt_vs_accuracies.append(iter_cnt_vs_accu)

iter_cnt_vs_accuracies_png = os.path.join(
    report_path, 'iter_cnt_vs_accuracies_{}.png'.format(max_iters))
title = 'Accuracy with Leave-Out-One w/ {} Iterations'.format(max_iters)
utils.draw_accuracies(iter_cnt_vs_accuracies, 'Iterations', 'Accuracy',
                      title, iter_cnt_vs_accuracies_png, legends)
