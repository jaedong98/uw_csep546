import collections

import model.DecisionTreeModel as dt
import model.LogisticRegressionModel as lg
import model.RandomForestsModel as rf


class BestSpamModel(object):
    """A model that predicts the most common label from the training data."""

    def __init__(self,
                 threshold=0.5,          # logistic regression
                 num_trees=40,            # random forest
                 bagging_w_replacement=False,      # random forest
                 feature_restriction=0,  # random forest
                 seed=10000):            # random forest
        self.lg = lg.LogisticRegressionModel(threshold=threshold)
        self.dt = dt.DecisionTreeModel()
        self.rf = rf.RandomForestModel(numTrees=num_trees,
                                       bagging_w_replacement=bagging_w_replacement,
                                       feature_restriction=feature_restriction,
                                       seed=seed)
        self.lg_pred = []
        self.dt_pred = []
        self.rf_pred = []

    def fit(self, xTrains, yTrains, iterations, min_to_stop):

        print("Fitting Logistic Regression with {} iterations"
              .format(iterations))
        self.lg.weights = [0.] * (len(xTrains[0]) + 1)
        xTrains_w_0 = [[1] + x for x in xTrains]
        self.lg.fit(xTrains_w_0, yTrains, iterations)

        print("Fitting Decision Tree Model with min to stop {}"
              .format(min_to_stop))
        self.dt.fit(xTrains, yTrains, min_to_stop)

        print("Fitting Random Forests with {} trees and min to stop {}"
              .format(self.rf.numTrees, min_to_stop))
        self.rf.fit(xTrains, yTrains, min_to_stop)

    def predict(self, xTests, threshold=None):
        xTests_w_0 = [[1] + x for x in xTests]
        self.lg_pred = self.lg.predict(xTests_w_0)
        self.dt_pred = self.dt.predict(xTests)
        self.rf_pred = self.rf.predict(xTests)

        predictions = []
        for l, d, r in zip(self.lg_pred, self.dt_pred, self.rf_pred):
            preds = [l, d, r]
            mc = collections.Counter(preds).most_common(1)[0][0]
            if threshold is None:
                predictions.append(mc)
            else:
                spam_percentage = preds.count(mc) / len(preds)
                predictions.append(int(spam_percentage <= threshold))

        return predictions

    def predict_probabilities(self, xTests):
        self.predict(xTests)
        predictions = []
        for l, d, r in zip(self.lg_pred, self.dt_pred, self.rf_pred):
            preds = [l, d, r]
            spam_percentage = preds.count(1) / len(preds)
            predictions.append(spam_percentage)

        return predictions
