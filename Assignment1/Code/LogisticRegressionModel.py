import math


class LogisticRegressionModel(object):
    """A logistic regression spam filter"""

    def __init__(self):
        pass

    def fit(self, xTrain, yTrain, iterations, step=0.01):
        self.weights = [.75, .75, .75, .25, .25]
        print("Fitting training dataset with {} iteration".format(iterations))
        print("Initial: {}".format(self.weights))
        while iterations > 0:
            yTrainPredicted = self.predict(xTrain)
            for i in range(len(self.weights)):
                partial_loss = 0
                j = 0
                for x, yhat_j, y_j in zip(xTrain, yTrainPredicted, yTrain):
                    partial_loss += (yhat_j - y_j) * x[i]
                    j += 1

                partial_derv = partial_loss / len(yTrain)
                self.weights[i] = self.weights[i] - step * partial_derv

            iterations -= 1

        print("UPDATED: {}".format(self.weights))

    def loss(self, xTest, yTest):

        yTestPredicted = self.predict(xTest)
        loss = 0
        for y_hat, y in zip(yTestPredicted, yTest):
            loss += (-y * math.log(y_hat)) - ((1 - y) * (math.log(1.0-y_hat)))
        return loss

    def predict(self, x):

        predictions = []

        for example in x:
            scores = [example[i] * self.weights[i]
                      for i in range(len(example))]
            z = self.weights[0] + sum(scores)
            prediction = 1.0 / (1.0 + math.exp(-z))
            predictions.append(prediction)
            # if prediction > 0.5:
            #    predictions.append(1)
            # else:
            #    predictions.append(0)

        return predictions
