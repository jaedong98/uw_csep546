import os

import Assignment3Support as utils
import EvaluationsStub
import DecisionTreeModel as dtm
# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


# Loading data
(xRaw, yRaw) = utils.LoadRawData(kDataPath)

# Train-Test split
(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrain, xTest) = utils.FeaturizeWNumericFeature(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

min_to_stop = 300
accuracy_md = os.path.join(report_path, 'prob2_part1_accuracy.md')
model = dtm.DecisionTreeModel()
model.fit(xTrain, yTrain, min_to_stop)
model.visualize()

yTestPredicted = model.predict(xTest)
accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
zn = 1.96
N = len(yTrain)

upper, lower = utils.calculate_bounds(accuracy, zn, N)
results = "* Before changing feature selections"
results += "\n  * Accuracy: {}, Lower: {}, Upper: {}".format(accuracy, lower, upper)
print(results)

min_to_stops = []
accuracies = []

start = 100
end = 1010
step = 10
for min_to_stop in [x for x in range(start, end, step)]:
    accuracy_md = os.path.join(report_path, 'prob2_part1_accuracy.md')
    model = dtm.DecisionTreeModel()
    model.fit(xTrain, yTrain, min_to_stop)
    model.visualize()

    yTestPredicted = model.predict(xTest)
    accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    zn = 1.96
    N = len(yTrain)

    upper, lower = utils.calculate_bounds(accuracy, zn, N)

    min_to_stops.append(min_to_stop)
    accuracies.append((lower, accuracy, upper))


img_fname = os.path.join(report_path,
                         'prob2_part2_accuracy_{}_{}_{}.png'
                         .format(start, end, step))

utils.draw_accuracies_vs_min_to_stps(min_to_stops,
                                     accuracies,
                                     'MinToStops',
                                     'Accuracies',
                                     'Accuracies vs. MinToStops',
                                     img_fname,
                                     ['Lower Bound', 'Accuracy Estimates', 'Upper Bound'])
