import math


class LogisticRegressionModel(object):
    """A logistic regression spam filter"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.w1_vs_iterations = []
        self.training_set_los_vs_iterations = []

    def fit(self, xTrain, yTrain, iterations, step=0.01):
        self.weights = [.05, .05, .05, .05, .05]
        print("Fitting training dataset with {} iteration".format(iterations))
        print("Initial: {}".format(self.weights))
        cnt = 0
        while cnt < iterations:
            yTrainPredicted = self.sigmoid(xTrain)
            for i in range(len(self.weights)):
                partial_loss = 0.0
                n = 0
                for x, yhat_j, y_j in zip(xTrain, yTrainPredicted, yTrain):
                    partial_loss += (yhat_j - y_j) * x[i]
                    n += 1

                partial_derv_loss = partial_loss / n
                self.weights[i] = self.weights[i] - step * partial_derv_loss
            
            if cnt % 10000:
                self.w1_vs_iterations.append((cnt, self.weights[1]))
            
            if cnt % 1000:
                training_loss = self.loss_calculator(yTrainPredicted, yTrain)
                self.training_set_los_vs_iterations.append((cnt, training_loss))

            cnt += 1

    def loss_calculator(self, yPredicted, ys):
        loss = 0
        for y_hat, y in zip(yPredicted, ys):
            loss += (-y * math.log(y_hat)) - ((1 - y) * (math.log(1.0-y_hat)))
        return loss

    def loss(self, xTest, yTest):

        yTestPredicted = self.sigmoid(xTest)
        return self.loss_calculator(yTestPredicted, yTest)

    def sigmoid(self, x):

        sigmoids = []

        for example in x:
            scores = [example[i] * self.weights[i]
                      for i in range(len(example))]
            z = self.weights[0] + sum(scores)
            sigmoid = 1.0 / (1.0 + math.exp(-z))
            sigmoids.append(sigmoid)

        return sigmoids

    def predict(self, x):

        sigmoids = self.sigmoid(x)
        predictions = []
        for s in sigmoids:
            if s > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions
