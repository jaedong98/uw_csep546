import itertools
import os

import utils.EvaluationsStub as ev
from Assignment4.Code import report_path, kDataPath
from model.BestSpamModel import BestSpamModel
from utils.Assignment4Support import table_for_accuracy_estimate_comparison
from utils.EvaluationsStub import Evaluation, EvaluateAll
from utils.data_loader import get_featurized_xs_ys


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


def calculate_accuracy_by_cv(config,
                             fname='',
                             k=5,
                             file_obj=None,
                             with_noise=True):
    """
    COPIED from homework2 but
    MODIFIED for homework3 as feature selection isn't necessary.

    Calculatge the accracy of cross validation using gradient descent.
    Returns: accuracy from cross validation.
    """
    fs_config = {'numFrequentWords': config['feature_selection_by_frequency'],
                 'numMutualInformationWords': config['feature_selection_by_mi'],
                 'includeHandCraftedFeatures': config['include_handcrafted_features'],
                 'with_noise': with_noise}

    xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(**fs_config)
    # folding data into Train vs Validation
    fold_xTrains, fold_xVals = fold_data(xTrain, k)
    fold_yTrains, fold_yVals = fold_data(yTrain, k)

    i = 0
    total_correct = 0
    status = '\nConfiguration:'
    for k, v in config.items():
        status += '\n * {}: {}'.format(k, v)
    for f_xTrain, f_xVal, f_yTrain, f_yVal in zip(fold_xTrains, fold_xVals,
                                                  fold_yTrains, fold_yVals):

        print("Cross validation for {}th folding".format(i))
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            bagging_w_replacement=config['bagging_w_replacement'],
                            feature_restriction=config['feature_restriction'])
        bsm.fit(f_xTrain, f_yTrain, config['iterations'], config['min_to_stop'])

        f_yVal_predict = bsm.predict(f_xVal)
        # compare and count corrections
        for p, v in zip(f_yVal_predict, f_yVal):
            if p == v:
                total_correct += 1
        status += '\n'
        status += "\nDecisionTreeModel for {}th folding".format(i)
        status += '\n'
        status += EvaluateAll(f_yVal, f_yVal_predict)
        status += '\n'
        print("Total correction so far: {}".format(total_correct))

        i += 1

    accuracy = total_correct / len(xTrain)
    print("Accuracy: {}".format(accuracy))
    print("Summary:")
    print(status)

    # accuracy on hold out data
    bsm = BestSpamModel(num_trees=config['num_trees'],
                        bagging_w_replacement=config['bagging_w_replacement'],
                        feature_restriction=config['feature_restriction'])
    bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
    yTestPredicted = bsm.predict(xTest)
    ev = Evaluation(yTest, yTestPredicted)
    comp_table = table_for_accuracy_estimate_comparison([accuracy, ev.accuracy],
                                                        ["Training Data", "Hold-out Data"],
                                                        N=len(xTrain))
    print(comp_table)

    if fname:
        with open(fname, 'w') as f:
            f.write("\nOverall Accuracy: {}".format(accuracy))
            f.write("\nAccuracy Estimate Comparison: ")
            f.write("\n{}".format(comp_table))
            f.write(status)

    if file_obj:
        file_obj.write("Overall Accuracy: {}\n".format(accuracy))
        file_obj.write(status)

    return accuracy


if __name__ == '__main__':
    config = {
        'name': 'Improved',
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 40,  # random forest
        'feature_restriction': 100,  # random forest
        'feature_selection_by_mi': 250,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0,  # 0 means False, N > 0 means select top N words based on frequency.
        'include_handcrafted_features': True
    }
    fname = os.path.join(report_path, 'prob2_cross_validation_accuracy.md')
    calculate_accuracy_by_cv(config, fname=fname, with_noise=True, k=5)
