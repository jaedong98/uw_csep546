import itertools
import os

import Assignment3Support as utils
import EvaluationsStub as ev
import DecisionTreeModel as dtm


# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def fold_data(xTrainRaw, k):
    """
    Args:
        xTrainRaw: a list of xTrainRaw data
        k: number of folding
    Returns:
        (a list of training sets, a list of validation sets)
    """
    if not k > 1:
        raise ValueError("Expected {} > 1".format(k))

    # divid xTrainRaw data into k group
    grouped_xTrainRaw = divide_into_group(xTrainRaw, k)
    print("Groupped xTrainRaw into {} groups.".format(k))

    trains = []
    validations = []
    for i in range(k):
        group = list(grouped_xTrainRaw)
        validation = group.pop(i)
        train = list(itertools.chain.from_iterable(group))
        trains.append(train)
        validations.append(validation)

    return trains, validations


def divide_into_group(xTrainRaw, k):

    cnt = len(xTrainRaw) / k
    groups = []
    last = 0.0

    while last < len(xTrainRaw):
        groups.append(xTrainRaw[int(last):int(last + cnt)])
        last += cnt

    if not len(xTrainRaw) == sum([len(g) for g in groups]):
        raise AssertionError("Missing/Duplicated element {} vs {}"
                             .format(len(xTrainRaw),
                                     sum([len(g) for g in groups])))

    if not len(groups) == k:
        raise AssertionError("More or less groups found {} vs (expected){}"
                             .format(len(groups), k))
    return groups


def calculate_accuracy_by_cv(xTrainRaw, yTrainRaw,
                             fname='',
                             k=5,
                             min_to_stop=100,
                             featurize=utils.FeaturizeWNumericFeature,
                             file_obj=None):
    """
    COPIED from homework2 but
    MODIFIED for homework3 as feature selection isn't necessary.

    Calculatge the accracy of cross validation using gradient descent.
    Returns: accuracy from cross validation.
    """

    # folding data into Train vs Validation
    fold_xTrains, fold_xVals = fold_data(xTrainRaw, k)
    fold_yTrains, fold_yVals = fold_data(yTrainRaw, k)

    i = 0
    total_correct = 0
    status = ''
    for f_xTrain, f_xVal, f_yTrain, f_yVal in zip(fold_xTrains, fold_xVals,
                                                  fold_yTrains, fold_yVals):

        # feature engineering on folded xTrain
        f_xTrain, _ = featurize(f_xTrain, [])
        f_xVal, _ = featurize(f_xVal, [])

        print("Cross validation for {}th folding".format(i))
        model = dtm.DecisionTreeModel()
        model.fit(f_xTrain, f_yTrain, min_to_stop)
        f_yVal_predict = model.predict(f_xVal)

        # compare and count corrections
        for p, v in zip(f_yVal_predict, f_yVal):
            if p == v:
                total_correct += 1
        status += '\n'
        status += "\nDecisionTreeModel for {}th folding".format(i)
        status += '\n'
        status += ev.EvaluateAll(f_yVal, f_yVal_predict)
        status += '\n'
        print("Total correction so far: {}".format(total_correct))

        i += 1

    accuracy = total_correct / len(xTrainRaw)
    print("Accuracy: {}".format(accuracy))
    print("Summary:")
    print(status)

    if fname:
        with open(fname, 'w') as f:
            f.write(status)

    if file_obj:
        file_obj.write(status)

    return accuracy


def process_cross_validation_with_min2stops(start, end, step, k, zn, N,
                                            report_path=report_path):

    cross_val_md = os.path.join(report_path,
                                'cross_val_{}_{}_{}_numeric_feature.md'
                                .format(start, end, step))

    with open(cross_val_md, 'w') as file_obj:
        min_to_stops = []
        accuracies = []
        best_accuracy = 0
        min_to_stop_at_best_accuracy = 0
        for min_to_stop in [x for x in range(start, end, step)]:
            accu = calculate_accuracy_by_cv(xTrainRaw, yTrainRaw,
                                            fname='',
                                            k=k,
                                            min_to_stop=min_to_stop,
                                            featurize=utils.FeaturizeWNumericFeature,
                                            file_obj=file_obj)
            upper, lower = utils.calculate_bounds(accu, zn, N)
            min_to_stops.append(min_to_stop)
            accuracies.append((lower, accu, upper))

            if accu > best_accuracy:
                best_accuracy = accu
                min_to_stop_at_best_accuracy = min_to_stop

        tunning_result = "* Best accuracy {} with MinToStop {}" \
            .format(best_accuracy, min_to_stop_at_best_accuracy)
        file_obj.write('\n')
        file_obj.write(tunning_result)

        img_fname = os.path.join(report_path,
                                 'prob2_part2_cross_val_accuracy_{}_{}_{}_numeric_feature.png'
                                 .format(start, end, step))

        utils.draw_accuracies_vs_min_to_stps(min_to_stops,
                                             accuracies,
                                             'MinToStops',
                                             'Accuracies',
                                             'Accuracies vs. MinToStops - Cross Validation',
                                             img_fname,
                                             ['Lower Bound', 'Accuracy Estimates', 'Upper Bound'])


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = utils.TrainTestSplit(xRaw,
                                                                      yRaw)

    start = 100
    end = 1010
    step = 10
    k = 5
    zn = 1.96
    N = len(xTrainRaw)
    process_cross_validation_with_min2stops(start, end, step, k, zn, N)
