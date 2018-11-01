import os
import utils.Assignment5Support as sup
from Assignment5.Code import kDataPath, report_path
from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation


def get_best(xTrain, yTrain, xTest, yTest, configs):
    best_accuracy = -1
    best_config = None
    best_ev = None
    for config in configs:
        model = RandomForestModel(numTrees=config['num_trees'],
                                  bagging_w_replacement=config['bagging_w_replacement'])
        model.fit(xTrain, yTrain, min_to_split=config['min_to_split'])
        yTestPredicted = model.predict(xTest)
        ev = Evaluation(yTest, yTestPredicted, indent=' '*6)
        if ev.accuracy > best_accuracy:
            best_accuracy = ev.accuracy
            best_config = config
            best_ev = ev
        for k, v in config.items():
            print("{}: {}".format(k, v))
        print(ev)

    print("*** Best accuracy: {}".format(best_accuracy))

    return best_config, best_ev


def accuracy_w_grid_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs):
    (xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, grid_y_gradients=True)
    yTrain = yTrainRaw
    yTest = yTestRaw

    return get_best(xTrain, yTrain, xTest, yTest, configs)


def accuracy_w_grid_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs):
    (xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, grid_x_gradients=True)
    yTrain = yTrainRaw
    yTest = yTestRaw

    return get_best(xTrain, yTrain, xTest, yTest, configs)


def accuracy_w_hist_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs):
    (xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, hist_y_gradients=True)
    yTrain = yTrainRaw
    yTest = yTestRaw

    return get_best(xTrain, yTrain, xTest, yTest, configs)


def accuracy_w_hist_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs):
    (xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, hist_x_gradients=True)
    yTrain = yTrainRaw
    yTest = yTestRaw

    return get_best(xTrain, yTrain, xTest, yTest, configs)


if __name__ == '__main__':
    (xRaw, yRaw) = sup.LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = sup.TrainTestSplit(xRaw, yRaw, percentTest=.25)

    report_md = os.path.join(report_path, "prob1_cv_features_accuracy_param_sweep.md")

    y_gradients = []
    x_gradients = []
    y_hist_gradients = []
    x_hist_gradients = []

    # param sweep - min_to_split
    min_to_split_configs = []
    for min_to_split in [2, 20, 50, 100]:
        config = {'min_to_split': min_to_split,
                  'bagging_w_replacement': True,
                  'num_trees': 20,
                  'feature_restriction': 100}
        min_to_split_configs.append(config)
    cf, ev = accuracy_w_grid_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         min_to_split_configs)
    y_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_grid_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         min_to_split_configs)

    x_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         min_to_split_configs)
    y_hist_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         min_to_split_configs)
    x_hist_gradients.append((ev.accuracy, cf, ev))

    # param sweep - num_trees
    num_trees_configs = []
    for num_trees in [1, 20, 40, 60]:
        config = {'min_to_split': 20,
                  'bagging_w_replacement': True,
                  'num_trees': num_trees,
                  'feature_restriction': 100}
        num_trees_configs.append(config)
    cf, ev = accuracy_w_grid_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         num_trees_configs)
    y_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_grid_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         num_trees_configs)
    x_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         num_trees_configs)
    y_hist_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         num_trees_configs)
    x_hist_gradients.append((ev.accuracy, cf, ev))

    # param sweep - feature_restriction
    feature_restriction_configs = []
    for feature_restriction in [5, 10, 20, 100]:
        config = {'min_to_split': 20,
                  'bagging_w_replacement': True,
                  'num_trees': 60,
                  'feature_restriction': feature_restriction}
        feature_restriction_configs.append(config)
    cf, ev = accuracy_w_grid_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         feature_restriction_configs)
    y_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_grid_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         feature_restriction_configs)
    x_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_y_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         feature_restriction_configs)
    y_hist_gradients.append((ev.accuracy, cf, ev))
    cf, ev = accuracy_w_hist_x_gradients(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                         feature_restriction_configs)
    x_hist_gradients.append((ev.accuracy, cf, ev))

    y_gradients.sort()
    x_gradients.sort()
    y_hist_gradients.sort()
    x_hist_gradients.sort()

    with open(report_md, 'w') as f:
        f.write("* Accuracies and Parameters Selected")

        # y-gradients
        f.write("\n  * 3x3 grid + y gradients")
        f.write("\n    * Best Accuracy: {}".format(y_gradients[0][0]))
        for k, v in y_gradients[0][1].items():
            f.write("\n    * {}: {}".format(k, v))
        f.write("\n{}".format(y_gradients[0][2]))

        # x-gradients
        f.write("\n  * 3x3 grid + x gradients")
        f.write("\n    * Best Accuracy: {}".format(x_gradients[0][0]))
        for k, v in x_gradients[0][1].items():
            f.write("\n    * {}: {}".format(k, v))
        f.write("\n{}".format(x_gradients[0][2]))

        # y hist gradients
        f.write("\n  * Histogram of a image y-gradients")
        f.write("\n    * Best Accuracy: {}".format(y_hist_gradients[0][0]))
        for k, v in y_hist_gradients[0][1].items():
            f.write("\n    * {}: {}".format(k, v))
        f.write("\n{}".format(y_hist_gradients[0][2]))

        # x hist gradients
        f.write("\n  * Histogram of a image y-gradients")
        f.write("\n    * Best Accuracy: {}".format(x_hist_gradients[0][0]))
        for k, v in x_hist_gradients[0][1].items():
            f.write("\n    * {}: {}".format(k, v))
        f.write("\n{}".format(x_hist_gradients[0][2]))

