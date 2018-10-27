import os

from Assignment4.Code import report_path
from model.BestSpamModel import BestSpamModel
from utils.Assignment4Support import draw_random_forest_accuracy_variances
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys

config = {
    'iterations': 10000,  # logistic regression
    'min_to_stop': 100,  # decision tree and random forest
    'feature_restriction': 20,  # random forest
    'use_bagging': True,  # random forest.
    'num_trees': 40,  # random forest
    'feature_restriction': 20,  # random forest
    'feature_selection_by_mi': 20,  # 0 means False, N > 0 means select top N words based on mi.
    'feature_selection_by_frequency': 0  # 0 means False, N > 0 means select top N words based on frequency.
}


def parameter_sweeps_by_min_to_stop(min_to_stops=[1, 5, 10, 50, 100],
                                    config=config,
                                    with_noise=True):

    accuracies = []
    test_accuracies = []
    for min_to_stop in min_to_stops:
        mi_words = config['feature_selection_by_mi']
        xTrain, xTest, yTrain, yTests = get_featurized_xs_ys(numMutualInformationWords=mi_words,
                                                             with_noise=with_noise)
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            use_bagging=config['use_bagging'],
                            feature_restriction=config['feature_restriction'])
        bsm.fit(xTrain, yTrain, config['iterations'], min_to_stop)
        # on hold-out data
        yTestsPredicted = bsm.predict(xTest)
        ev = Evaluation(yTests, yTestsPredicted)
        print(ev)
        accuracies.append((min_to_stop, ev.accuracy))

        # on training data to show over-/under-fitting
        yTrainPredicted = bsm.predict(xTrain)
        ev = Evaluation(yTrain, yTrainPredicted)
        test_accuracies.append((min_to_stop, ev.accuracy))

    xlabel = 'MinToSplit'
    ylabel = 'Accuracies'
    title = 'Best SMS Spam Model (with noise)'
    fname = 'prob2_param_sweep_by_min_to_split_{}.png' \
        .format('_'.join([str(n) for n in min_to_stops]))
    img_fname = os.path.join(report_path, fname)
    legends = ['Test Data', 'Training Data']

    draw_random_forest_accuracy_variances([accuracies, test_accuracies],
                                          xlabel, ylabel,
                                          title,
                                          img_fname,
                                          legends=legends)

    return accuracies, test_accuracies


def parameter_sweeps_by_feature_restriction(feature_restrictions=[10, 50, 100, 150, 250],
                                            config=config,
                                            with_noise=True):

    accuracies = []
    test_accuracies = []
    for feature_restriction in feature_restrictions:
        mi_words = config['feature_selection_by_mi']
        xTrain, xTest, yTrain, yTests = get_featurized_xs_ys(numMutualInformationWords=mi_words,
                                                             with_noise=with_noise)
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            use_bagging=config['use_bagging'],
                            feature_restriction=feature_restriction)
        bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
        # on hold-out data
        yTestsPredicted = bsm.predict(xTest)
        ev = Evaluation(yTests, yTestsPredicted)
        print(ev)
        accuracies.append((feature_restriction, ev.accuracy))

        # on training data to show over-/under-fitting
        yTrainPredicted = bsm.predict(xTrain)
        ev = Evaluation(yTrain, yTrainPredicted)
        test_accuracies.append((feature_restriction, ev.accuracy))

    xlabel = 'Feature Restrictions'
    ylabel = 'Accuracies'
    title = 'Best SMS Spam Model (with noise)'
    fname = 'prob2_param_sweep_by_feature_restriction_{}.png' \
        .format('_'.join([str(n) for n in feature_restrictions]))
    img_fname = os.path.join(report_path, fname)
    legends = ['Test Data', 'Training Data']

    draw_random_forest_accuracy_variances([accuracies, test_accuracies],
                                          xlabel, ylabel,
                                          title,
                                          img_fname,
                                          legends=legends)

    return accuracies, test_accuracies


def parameter_sweeps_by_mi(mi_words=[20, 50, 100, 200, 250],
                                    config=config,
                                    with_noise=True):

    accuracies = []
    test_accuracies = []
    for mi_word in mi_words:
        xTrain, xTest, yTrain, yTests = get_featurized_xs_ys(numMutualInformationWords=mi_word,
                                                             with_noise=with_noise)
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            use_bagging=config['use_bagging'],
                            feature_restriction=config['feature_restriction'])
        bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
        # on hold-out data
        yTestsPredicted = bsm.predict(xTest)
        ev = Evaluation(yTests, yTestsPredicted)
        print(ev)
        accuracies.append((mi_word, ev.accuracy))

        # on training data to show over-/under-fitting
        yTrainPredicted = bsm.predict(xTrain)
        ev = Evaluation(yTrain, yTrainPredicted)
        test_accuracies.append((mi_word, ev.accuracy))

    xlabel = 'N features selected by MI'
    ylabel = 'Accuracies'
    title = 'Best SMS Spam Model (with noise)'
    fname = 'prob2_param_sweep_by_mi_{}.png' \
        .format('_'.join([str(n) for n in mi_words]))
    img_fname = os.path.join(report_path, fname)
    legends = ['Test Data', 'Training Data']

    draw_random_forest_accuracy_variances([accuracies, test_accuracies],
                                          xlabel, ylabel,
                                          title,
                                          img_fname,
                                          legends=legends)

    return accuracies, test_accuracies


if __name__ == '__main__':
    config = {
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'feature_restriction': 100,  # random forest
        'use_bagging': True,  # random forest.
        'num_trees': 10,  # random forest
        'feature_restriction': 20,  # random forest
        'feature_selection_by_mi': 20,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 10  # 0 means False, N > 0 means select top N words based on frequency.
    }
    parameter_sweeps_by_mi(config=config, with_noise=True)
    parameter_sweeps_by_min_to_stop(config=config, with_noise=True)
    parameter_sweeps_by_feature_restriction(config=config,
                                            with_noise=True)
