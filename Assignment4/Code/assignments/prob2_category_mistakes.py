import os

import utils.EvaluationsStub as es
from Assignment4.Code import report_path

from model.BestSpamModel import BestSpamModel
from utils.data_loader import get_featurized_xs_ys, get_xy_test_raw


def find_categrize_mistakes(config, with_noise=True, top=20):

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
    yTestPredicted = bsm.predict(xTest)

    fn = []
    fp = []

    for i, (t, p) in enumerate(zip(yTest, yTestPredicted)):
        prob = yTestPredicted_prob[i]
        if (t, p) == (1, 0):  # false negative
            fn.append((prob, i))
        elif (t, p) == (0, 1):  # false positive
            fp.append((prob, i))

    # the true answer was 1, but the model gives very low probabilities
    sorted_fn = sorted(fn)
    # the true answer was 0, but gives very high probabilities.
    sorted_fp = sorted(fp, reverse=True)
    print('*' * 80)
    print('Total {} false netagives.'.format(len(sorted_fn)))
    print('Total {} false positives.'.format(len(sorted_fp)))
    print('False Netagives: {}'.format(sorted_fn))
    print('False Positives: {}'.format(sorted_fp))
    print(es.ConfusionMatrix(yTest, yTestPredicted))
    print('*' * 80)
    return sorted_fn[:top], sorted_fp[:top]


def generate_mistakes_table(mistakes, title, header, xTestRaw, fname, w=30):

    top_mistakes = title
    top_mistakes += '\n'
    top_mistakes += '\n  {}'.format(header)
    top_mistakes += '\n  |-|-|'
    for prob, i in mistakes:
        top_mistakes += '\n  |{}| {}|'.format(
            '{}'.format(prob).center(w), xTestRaw[i].strip())

    with open(fname, 'w') as f:
        f.write(top_mistakes)
    print("Created {}".format(fname))


if __name__ == '__main__':
    config = {
        'iterations': 10000,  # logistic regression
        'min_to_stop': 2,  # decision tree and random forest
        'feature_restriction': 20,  # random forest
        'bagging_w_replacement': True,  # random forest.
        'num_trees': 20,  # random forest
        'feature_restriction': 20,  # random forest
        'feature_selection_by_mi': 20,  # 0 means False, N > 0 means select top N words based on mi.
        'feature_selection_by_frequency': 0  # 0 means False, N > 0 means select top N words based on frequency.
    }

    ############################################################################
    # by mutual information
    sorted_fn, sorted_fp = find_categrize_mistakes(config)
    xTestRaw, yTestRaw = get_xy_test_raw(with_noise=True)
    yTest = yTestRaw

    w = 30
    header = '| Probabilities | Test Raw |'
    # ------------------------------------------------------------------------ #
    title = '* False Positive - the true answer was 0, but gives very high probabilities'
    fname = os.path.join(report_path, 'prob2_category_mistake_false_positives.md')

    generate_mistakes_table(sorted_fp, title, header, xTestRaw, fname)

    title = '* False Negative - the true answer was 1, but gives very low probabilities'
    fname = os.path.join(report_path, 'prob2_category_mistake_false_negative.md')

    generate_mistakes_table(sorted_fn, title, header, xTestRaw, fname)

