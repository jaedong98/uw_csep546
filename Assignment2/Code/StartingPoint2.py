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

def calculate_accuracy(xTrainRaw, xTestRaw, yTrain, feature_selection_methods):
    xTrain = utils.FeaturizeTraining(xTrainRaw, feature_selection_methods)
    xTest = utils.FeaturizeTraining(xTestRaw, feature_selection_methods)
    model = lgm.LogisticRegressionModel(initial_w0=0.0,
                                        initial_weights=[0.0] * len(feature_selection_methods))

    import SpamHeuristicModel
    model = SpamHeuristicModel.SpamHeuristicModel()
    model.fit(xTrain, yTrain)
    yTestPredicted = model.predict(xTest)

    return EvaluationsStub.Accuracy(yTest, yTestPredicted)

tic = time.time()
accuracies = []
legends = []
for i in range(len(feature_selection_methods_options)):
    feature_selection_methods = list(feature_selection_methods_options)
    function_out = feature_selection_methods.pop(i)
    legends.append("w/o {}".format(function_out.__name__.upper()))
    print("Exclude {}".format(function_out.__name__))
    
    accuracy = calculate_accuracy(xTrainRaw, 
                                          xTestRaw,
                                          yTrain,
                                          feature_selection_methods)

    accuracies.append(accuracy)

legends.append("w/ All of Features")
accuracy = calculate_accuracy(xTrainRaw, xTestRaw, yTrain, feature_selection_methods_options)
accuracies.append(accuracy)

table = utils.accuracy_table(accuracies, legends)
table_md = os.path.join(report_path, 'accuracy_table_w_heuristic.md')

print(table)

with open(table_md, 'w') as f:
    f.write(table)
    f.writelines("\n")
    f.writelines("\nTook: {} sec.".format(time.time() - tic))