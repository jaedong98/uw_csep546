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

# create a picture accuracy vs min_to_stops
min_to_stops = []
accuracies = []
start = 100
end = 1010
step = 10
best_accuracy = 0
min_to_stop_at_best_accuracy = 0

model = dtm.DecisionTreeModel()
for min_to_stop in [x for x in range(start, end, step)]:

    model.fit(xTrain, yTrain, min_to_stop)
    model.visualize()

    yTestPredicted = model.predict(xTest)
    accuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)
    zn = 1.96
    N = len(yTrain)

    upper, lower = utils.calculate_bounds(accuracy, zn, N)

    min_to_stops.append(min_to_stop)
    accuracies.append((lower, accuracy, upper))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        min_to_stop_at_best_accuracy = min_to_stop


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

tunning_result = "* Best accuracy {} with MinToStop {}"\
    .format(best_accuracy, min_to_stop_at_best_accuracy)
print(tunning_result)

tuning_md = os.path.join(report_path, 'prob2_part2_tuning_min_to_stop.md')
with open(tuning_md, 'w') as f:
    f.write(tunning_result)
    f.write("\n* Model visualization with min to stop {}"
            .format(min_to_stop_at_best_accuracy))
    model.fit(xTrain, yTrain, min_to_stop_at_best_accuracy)
    model.visualize(f)

# ROC curve comparision (False Positive Rate vs False Negative Rate)

