import os

import utils.Assignment3Support as sup
import utils.EvaluationsStub
import models.DecisionTreeModel as dtm
# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


# Loading data
(xRaw, yRaw) = sup.LoadRawData(kDataPath)

# Train-Test split
(xTrainRaw, yTrainRaw, xTestRaw,
 yTestRaw) = sup.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw)))

(xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw


# part 1
def generate_accuray_report(xTrain, yTrain, xTest, yTest, min_to_stop, report_path=report_path):
    accuracy_md = os.path.join(report_path,
                               'prob2_part1_accuracy_m2s{}.md'.format(min_to_stop))
    model = dtm.DecisionTreeModel()
    model.fit(xTrain, yTrain, min_to_stop)
    model.visualize()

    yTestPredicted = model.predict(xTest)
    accuracy = utils.EvaluationsStub.Accuracy(yTest, yTestPredicted)
    zn = 1.96
    N = len(yTrain)

    upper, lower = sup.calculate_bounds(accuracy, zn, N)
    results = "* Before changing feature selections"
    results += "\n  * Accuracy: {}, Lower: {}, Upper: {}".format(accuracy, lower, upper)
    print(results)

    with open(accuracy_md, 'w') as file_obj:
        file_obj.write(results)
        file_obj.write('\n  * Decision Tree with minToStop={}'.format(min_to_stop))
        model.visualize(file_obj)


if __name__ == "__main__":
    generate_accuray_report(xTrain, yTrain, xTest, yTest, min_to_stop=100)


