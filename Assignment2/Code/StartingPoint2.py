import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import Assignment2Support as utils
import EvaluationsStub

# UPDATE this path for your environment
import os
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

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

for i in range(len(feature_selection_methods_options)):
    feature_selection_methods = list(feature_selection_methods_options)
    function_out = feature_selection_methods.pop(i)
    print("Exclude {}".format(function_out.__name__))
    xTrain = utils.FeaturizeTraining(xTrainRaw, feature_selection_methods)


############################
print("#############################")
print("### Logistic regression model")


def draw_single_plot(tuples, xlabel, ylabel, title, img_fname):
    t, s = zip(*tuples)
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def draw_weights(iter_cnts, weights, xlabel, ylabel, title, img_fname):
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for ws in zip(*weights):
        ax.plot(iter_cnts, ws)

    ax.legend(('w0', 'w1', 'w2', 'w3', 'w4', 'w5'))
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

#w1_vs_iterations = []
max_iters = 50000
tic = time.time()
w1_vs_iters = []
test_loss_vs_iters = []
test_accuracy_vs_iters = []
training_set_loss_vs_iters = []
weights = [list(model.weights)]
iter_cnts = [0]
iter_step = 1000

# Extend xTrains and xTest with 1 at [0]
xTrain = [[1] + x for x in xTrain]
xTest = [[1] + x for x in xTest]

for i, iters in enumerate([iter_step] * 50):
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
    w1_vs_iters.append((iter_cnt, model.weights[1]))
    test_loss_vs_iters.append((iter_cnt, test_loss))
    test_accuracy_vs_iters.append((iter_cnt, test_accuracy))
    training_set_loss_vs_iters.append((iter_cnt, model.training_loss))
    weights.append(list(model.weights))
    iter_cnts.append(iter_cnt)

weights_png = os.path.join(report_path, 'weights_{}.png'.format(max_iters))
draw_weights(iter_cnts, weights, 'Iterations',
             'Weights', 'Weights', weights_png)

w1_png = os.path.join(report_path, 'w1_{}.png'.format(max_iters))
draw_single_plot(w1_vs_iters, 'Iterations', 'Weight[1]', 'Weight[1]s', w1_png)

test_loss_png = os.path.join(report_path, 'test_loss_{}.png'.format(max_iters))
draw_single_plot(test_loss_vs_iters, 'Iterations',
                 'Test loss', 'Test Loss', test_loss_png)

test_accuracy_vs_iters_png = os.path.join(
    report_path, 'test_accuracy_vs_iters_{}.png'.format(max_iters))
draw_single_plot(test_accuracy_vs_iters, 'Iterations',
                 'Test Accuracy', 'Test Accuracy', test_accuracy_vs_iters_png)

training_set_loss_png = os.path.join(
    report_path, 'training_set_loss_{}.png'.format(max_iters))
draw_single_plot(training_set_loss_vs_iters, 'Iterations',
                 'Training Set Loss', 'Training Set Loss', training_set_loss_png)
print("++++++++++++++++++++++++++++++++")

statistic_md = os.path.join(report_path, 'statictics.md')
results = EvaluationsStub.EvaluateAll(yTest, yTestPredicted)

with open(statistic_md, 'w') as f:
    f.write(results)
    f.writelines("\n")
    f.writelines("\n{} iterations".format(max_iters))
    f.writelines("\nTook: {} sec.".format(time.time() - tic))
