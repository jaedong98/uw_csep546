import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import Assignment1Support
import EvaluationsStub

# UPDATE this path for your environment
import os
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### 'Most Common' model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

print("### Heuristic model")

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

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
    

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

#w1_vs_iterations = []
max_iters = 50000
tic = time.time()
loss_vs_iters = []
test_accuracy_vs_iters = []
for iters in [10000, 20000, 30000, 40000, max_iters]:
    fit_tic = time.time()
    model.fit(xTrain, yTrain, iterations=iters, step=0.01)
    fit_toc = time.time() - fit_tic
    print("Took {} sec. Fitted data with {} iterations".format(fit_toc, iters))
    yTestPredicted = model.predict(xTest)
    test_loss = model.loss(xTest, yTest)
    test_accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    print("%d, %f, %f, %f" % (iters, model.weights[1], test_loss, test_accuracy))
    loss_vs_iters.append((iters, test_loss))
    test_accuracy_vs_iters.append((iters, test_accuracy))

w1_png = os.path.join(report_path, 'w1_{}.png'.format(max_iters))
draw_single_plot(model.w1_vs_iterations, 'Iterations', 'Weight[1]', 'Weight[1]s', w1_png)

test_loss_png = os.path.join(report_path, 'test_loss_{}.png'.format(max_iters))
draw_single_plot(loss_vs_iters, 'Iterations', 'Test loss', 'Test Loss', test_loss_png)

test_accuracy_vs_iters_png = os.path.join(report_path, 'test_accuracy_vs_iters_{}.png'.format(max_iters))
draw_single_plot(test_accuracy_vs_iters, 'Iterations', 'Test Accuracy', 'Test Accuracy', test_accuracy_vs_iters_png)

training_set_loss_png = os.path.join(report_path, 'training_set_loss_{}.png'.format(max_iters))
draw_single_plot(model.training_set_loss_vs_iterations, 'Iterations', 'Training Set Loss', 'Training Set Loss', training_set_loss_png)
print("++++++++++++++++++++++++++++++++")

statistic_md = os.path.join(report_path, 'statictics.md')
results = EvaluationsStub.EvaluateAll(yTest, yTestPredicted)

with open(statistic_md, 'w') as f:
    f.write(results)
    f.writelines("\n")
    f.writelines("\n{} iterations".format(iters))
    f.writelines("\nTook: {} sec.".format(time.time() - tic))
