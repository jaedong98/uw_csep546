import inspect
import os

import utils.Assignment4Support as sup
import utils.EvaluationsStub as es
import model.DecisionTreeModel as dtm
from Assignment4.Code import report_path
from model.BestSpamModel import BestSpamModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys

"""
What is an ROC curve?

Ans. plot ( sensitivity vs (1 - specificity ) ) !!

Let's assume, you have built a Logistic Regression model.

1. while predicting, you need to give a threshold and based on that you'll get the predicted output and from that you can calculate sensitivity & specificity.
2. Now, go back to the predicting step and give some 10 threshold values from 0 to 1.
3. So, you have 10 sensitivity & specificity values!!
4. Arrange them in the increasing order of (1-specificity).
5. Draw the plot using these values.
"""


def get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                   min_to_step,
                   threshold,
                   featurize=sup.Featurize):

    yTrain = yTrainRaw
    yTest = yTestRaw

    (xTrain, xTest) = featurize(xTrainRaw, xTestRaw)

    model = dtm.DecisionTreeModel()
    model.fit(xTrain, yTrain, min_to_step)

    yTestPredicted = model.predict(xTest, threshold)
    return es.Evaluation(yTest, yTestPredicted)


def compare_roc_curves_by_configs(configs,
                                  thresholds,
                                  report_path=report_path,
                                  with_noise=True):
    graphs = []
    legends = []
    original_fpr_fnr = []
    for i, config in enumerate(configs):
        legends.append('Config {}'.format(config['name']))
        fs_config = {'numFrequentWords': config['feature_selection_by_frequency'],
                     'numMutualInformationWords': config['feature_selection_by_mi'],
                     'includeHandCraftedFeatures': False,
                     'with_noise': with_noise}
        xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(**fs_config)
        print("Loaded all data.")
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            bagging_w_replacement=config['bagging_w_replacement'],
                            feature_restriction=config['feature_restriction'])
        bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
        yTestPredicted_prob = bsm.predict_probabilities(xTest)

        cont_length_fpr_fnr = []
        for threshold in thresholds:
            yTestPredicted = [int(p >= threshold) for p in yTestPredicted_prob]
            #featured = [int(y < threshold) for y in yTestPredicted]
            ev = Evaluation(yTest, yTestPredicted)
            #if threshold == 0:
            #    assert (ev.fpr == 0.0), ev
            print(ev)
            cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
        graphs.append(cont_length_fpr_fnr)

    # plotting
    start = thresholds[0]
    end = thresholds[-1]
    step = thresholds[1] - thresholds[0]
    fname = '{}_{}_{}_{}.png'.format(inspect.stack()[0][3], start, end, step)
    img_fname = os.path.join(report_path, fname)
    sup.draw_accuracies(graphs,
                          'False Positive Rate', 'False Negative Rate', '',
                          img_fname,
                          legends=legends,
                          invert_yaxis=True)


if __name__ == '__main__':
    config_baseline = {
        'name': 'Baseline',
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 20,  # random forest
        'feature_restriction': 20,  # random forest
        'feature_selection_by_mi': 20,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0,  # 0 means False, N > 0 means select top N words based on frequency.
        'include_handcrafted_features': True
    }
    config_improved = {
        'name': 'Improved',
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 20,  # random forest
        'feature_restriction': 20,  # random forest
        'feature_selection_by_mi': 100,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0,  # 0 means False, N > 0 means select top N words based on frequency.
        'include_handcrafted_features': True
    }
    config_best = {
        'name': 'Best',
        'iterations': 10000,  # logistic regression
        'min_to_stop': 100,  # decision tree and random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 20,  # random forest
        'feature_restriction': 100,  # random forest
        'feature_selection_by_mi': 100,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0,  # 0 means False, N > 0 means select top N words based on frequency.
        'include_handcrafted_features': True
    }
    start = 0
    end = 1
    N = 100
    thresholds = [x / N for x in range(N + 1)]
    configs = [config_best, config_improved, config_baseline]
    compare_roc_curves_by_configs(configs, thresholds)