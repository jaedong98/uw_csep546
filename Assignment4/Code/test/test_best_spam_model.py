import unittest

from model.BestSpamModel import BestSpamModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys

config = {
    'iterations': 40000,  # logistic regression
    'min_to_stop': 100,  # decision tree and random forest
    'feature_restriction': 20,  # random forest
    'use_bagging': True,  # random forest.
    'num_trees': 40,  # random forest
    'feature_restriction': 20,  # random forest
    'feature_selection_by_mi': 0,  # 0 means False, N > 0 means select top N words based on mi.
    'feature_selection_by_frequency': 0  # 0 means False, N > 0 means select top N words based on frequency.
}


class TestBestSpamModel(unittest.TestCase):

    def test_fit_predict(self):
        xTrain, xTest, yTrain, yTests = get_featurized_xs_ys()
        bsm = BestSpamModel(num_trees=config['num_trees'],
                            use_bagging=config['use_bagging'],
                            feature_restriction=config['feature_restriction'])
        config['iterations'] = 10
        bsm.fit(xTrain, yTrain, config['iterations'], config['min_to_stop'])
        yTestsPredicted = bsm.predict(xTest)

        print(Evaluation(yTests, yTestsPredicted))


if __name__ == '__main__':
    unittest.main()
