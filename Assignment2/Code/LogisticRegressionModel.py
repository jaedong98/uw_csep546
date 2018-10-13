from collections import OrderedDict
import math
import numpy as np


class LogisticRegressionModel(object):
    """A logistic regression spam filter"""

    def __init__(self,
                 threshold=0.5,
                 cnt_to_log=[],
                 initial_weights=[.0, .0, .0, .0, .0],
                 initial_w0=.0):
        self.threshold = threshold
        self.weights_logs = OrderedDict()
        self.cnt_to_log = cnt_to_log
        self.weights = [initial_w0] + initial_weights
        self.training_loss = 0

    def fit(self, xTrain, yTrain, iterations, step=0.01):

        if len(self.weights) != len(xTrain[0]):
            raise ValueError("Not aligned data. We assume feature vector to "
                             "include x0 = 1. {} vs {}".format(
                                 len(self.weights), len(xTrain[0])))

        print("Fitting training dataset with {} iteration".format(iterations))
        print("Initial: {}".format(self.weights))
        cnt = 0
        n = len(xTrain)
        dot = np.dot
        array = np.array
        weights_descents = []
        while cnt < iterations:
            yTrainPredicted = self.calculate_yhats(xTrain)
            ys_delta = array(yTrainPredicted) - array(yTrain)
            weights_descents.append([step * dot(ys_delta, xs) / n for xs in zip(*xTrain)])
            cnt += 1

        n_weights = []
        for w, w_des in zip(self.weights, zip(*weights_descents)):
            n_weights.append(w - sum(w_des))
        self.weights = n_weights
        self.training_loss = self.loss_calculator(yTrainPredicted, yTrain)


    def loss_calculator(self, yPredicted, ys):
        log = math.log
        return sum([log(1.0 - y_hat) if y == 0 else -y * log(y_hat) for y_hat, y in zip(yPredicted, ys)])


    def loss(self, xTest, yTest):

        yTestPredicted = self.calculate_yhats(xTest)
        return self.loss_calculator(yTestPredicted, yTest)

    def calculate_yhats(self, x):

        exp = math.exp
        dot = np.dot
        return [1.0 / (1.0 + exp(-dot(d , self.weights))) for d in x]

    def predict(self, x):

        yhats = self.calculate_yhats(x)
        return [1 if s > self.threshold else 0 for s in yhats]

    def predict_probabilities(self, x):
        return self.calculate_yhats(x)
