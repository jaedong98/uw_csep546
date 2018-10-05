
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
import LogisticRegressionModel
model = LogisticRegressionModel.LogisticRegressionModel()
print("#############################")
print("### Logistic regression model")
training_set_loss_vs_iterations = []
test_set_loss_vs_iterations = []
test_set_accuracy_vs_iterations = []
w1_vs_iterations = []
for i in [100, 200, 300, 400, 500]:
    model.fit(xTrain, yTrain, iterations=i, step=0.01)
    training_set_loss_vs_iterations.append((model.loss(xTrain, yTrain), i))

    yTestPredicted = model.predict(xTest)
    test_set_loss_vs_iterations.append((model.loss(xTest, yTest), i))
    test_set_accuracy_vs_iterations.append((EvaluationsStub.Accuracy(yTest, yTestPredicted), i))
    w1_vs_iterations.append((model.weights[1], i))

    print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(
          xTest, yTest), EvaluationsStub.Accuracy(yTest, yTestPredicted)))

print("Training set loss vs iterations: {}".format(training_set_loss_vs_iterations))
print("Test set loss vs iterations: {}".format(test_set_loss_vs_iterations))
print("Test set accuracy vs. iterations: {}".format(test_set_accuracy_vs_iterations))
print("W1 vs. iterations: {}".format(w1_vs_iterations))

print("++++++++++++++++++++++++++++++++")
EvaluationsStub.ExecuteAll(yTest, yTestPredicted)
