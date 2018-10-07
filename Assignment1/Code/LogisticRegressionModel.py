from collections import OrderedDict
import math
import numpy as np


class LogisticRegressionModel(object):
    """A logistic regression spam filter"""

    def __init__(self, threshold=0.5, cnt_to_log=[]):
        self.threshold = threshold
        self.weights_logs = OrderedDict()
        self.cnt_to_log = cnt_to_log
        

    def fit(self, xTrain, yTrain, iterations, step=0.01):
        self.weights = [.05, .05, .05, .05, .05]
        self.w1_vs_iterations = []
        self.training_set_loss_vs_iterations = []
        print("Fitting training dataset with {} iteration".format(iterations))
        print("Initial: {}".format(self.weights))
        cnt = 0
        n = len(xTrain)
        while cnt < iterations:
            yTrainPredicted = self.calculate_yhats(xTrain)
            for i, xs in enumerate(zip(*xTrain)):
                ys_delta = np.array(yTrainPredicted) - np.array(yTrain)
                partial_loss = np.dot(ys_delta, xs)
                partial_derv_loss = partial_loss / n
                self.weights[i] = self.weights[i] - step * partial_derv_loss
            
            if cnt % 10000:
                self.w1_vs_iterations.append((cnt, self.weights[1]))
            
            if cnt % 1000:
                training_loss = self.loss_calculator(yTrainPredicted, yTrain)
                self.training_set_loss_vs_iterations.append((cnt, training_loss))

            cnt += 1

    def loss_calculator(self, yPredicted, ys):
        loss = 0
        for y_hat, y in zip(yPredicted, ys):
            loss += (-y * math.log(y_hat)) - ((1 - y) * (math.log(1.0-y_hat)))
        return loss

    def loss(self, xTest, yTest):

        yTestPredicted = self.calculate_yhats(xTest)
        return self.loss_calculator(yTestPredicted, yTest)

    def calculate_yhats(self, x):

        sigmoids = []

        for example in x:
            scores = [example[i] * self.weights[i]
                      for i in range(len(example))]
            z = self.weights[0] + sum(scores)
            sigmoid = 1.0 / (1.0 + math.exp(-z))
            sigmoids.append(sigmoid)

        return sigmoids

    def predict(self, x):

        yhats = self.calculate_yhats(x)
        predictions = []
        for s in yhats:
            if s > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions
