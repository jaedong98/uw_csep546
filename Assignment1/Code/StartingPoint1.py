import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    

img_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()

#w1_vs_iterations = []
for i in [50000]:
    model.fit(xTrain, yTrain, iterations=i, step=0.01)
    yTestPredicted = model.predict(xTest)
    
    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(
          xTest, yTest), EvaluationsStub.Accuracy(yTest, yTestPredicted)))
w1_png = os.path.join(img_path, 'w1.png')
draw_single_plot(model.w1_vs_iterations, 'Iterations', 'Weight[1]', 'Weight[1]s', w1_png)

training_set_loss_png = os.path.join(img_path, 'training_set_loss.png')
draw_single_plot(model.training_set_los_vs_iterations, 'Iterations', 'Training Set Loss', 'Training Set Loss', training_set_loss_png)
print("++++++++++++++++++++++++++++++++")
EvaluationsStub.ExecuteAll(yTest, yTestPredicted)
