import math


class LogisticRegressionModel(object):
    """A logistic regression spam filter"""

    def __init__(self):
        pass

    def fit(self, xTrain, yTrain, iterations, step=0.01):
        self.weights = [.75, .75, .75, .25, .25]
        yPredicted = self.predict(xTrain)
        # update weight here for iterations

    def loss(xTest, yTest):

        # sum of (-y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i])))
        pass

    def predict(self, x):

        predictions = []

        for example in x:
            scores = [example[i] * self.weights[i]
                      for i in range(len(example))]
            z = self.weights[0] + sum(scores)
            prediction = 1.0 / 1.0 + math.exp(-z)
            if prediction > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions
