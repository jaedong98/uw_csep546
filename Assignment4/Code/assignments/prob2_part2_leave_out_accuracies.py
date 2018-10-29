import os

from Assignment4.Code import report_path
from assignments.prob2_category_mistakes_report_utils import many_uppers, has_url
from model.BestSpamModel import BestSpamModel
from utils.Assignment4Support import is_longger, has_number, contain_call, contain_to, contain_your, accuracy_table
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


def leave_out_accuracies(config, with_noise=True):
    feature_selection_methods_options = [is_longger,
                                         has_number,
                                         contain_call,
                                         contain_to,
                                         contain_your,
                                         many_uppers,
                                         has_url]

    accuracies = []
    legends = []
    for i in range(len(feature_selection_methods_options)):
        feature_selection_methods = list(feature_selection_methods_options)
        function_out = feature_selection_methods.pop(i)
        legends.append("w/o {}".format(function_out.__name__.upper()))
        print("Exclude {}".format(function_out.__name__))

        fs_config = {'numFrequentWords': config['feature_selection_by_frequency'],
                     'numMutualInformationWords': config['feature_selection_by_mi'],
                     'includeHandCraftedFeatures': feature_selection_methods_options,
                     'with_noise': with_noise}
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(**fs_config)
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            bagging_w_replacement=config['bagging_w_replacement'],
                            feature_restriction=config['feature_restriction'])
        bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])

        yTestPredicted = bsm.predict(xTest)
        ev = Evaluation(yTest, yTestPredicted)

        accuracies.append(ev.accuracy)

    legends.append("w/ All of Features")
    fs_config['includeHandCraftedFeatures'] = True
    xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(**fs_config)
    bsm = BestSpamModel(num_trees=config['num_trees'],
                        bagging_w_replacement=config['bagging_w_replacement'],
                        feature_restriction=config['feature_restriction'])
    bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
    yTestPredicted = bsm.predict(xTest)
    ev = Evaluation(yTest, yTestPredicted)
    accuracies.append(ev.accuracy)

    table = accuracy_table(accuracies, legends)
    table_md = os.path.join(report_path, 'prob2_leave_out_accuracy_table.md')

    print(table)

    with open(table_md, 'w') as f:
        f.write(table)


if __name__ == '__main__':
    config = {
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 20,  # random forest
        'feature_restriction': 20,  # random forest
        'feature_selection_by_mi': 100,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0  # 0 means False, N > 0 means select top N words based on frequency.
    }
    leave_out_accuracies(config)