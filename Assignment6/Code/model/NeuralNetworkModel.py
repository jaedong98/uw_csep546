"""
References:
    https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
    https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

"""
import numpy as np
np.random.seed(100)
LOCAL = False

# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, xTrains, yTrains, num_hidden_layer=2, num_nodes=10,
                 step_size=0.05):
        self.input = np.array(xTrains)  # should be flattened!
        self.y = np.array(yTrains)
        self.num_hidden_layer = num_hidden_layer
        self.num_nodes = num_nodes
        self.step_size = step_size  #0.05
        self.output = np.zeros(self.y.shape)
        self.weights = []
        self.get_initial_weights()
        self.outputs = []

    def get_initial_weights(self):

        for i in range(self.num_hidden_layer):

            if i == 0:
                num_features = len(self.input[0])
                w_dim = (num_features, self.num_nodes)
            else:
                w_dim = (self.num_nodes, self.num_nodes)

            if LOCAL:
                self.weights.append(np.random.rand(*w_dim))
            else:
                self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))

        w_dim = (self.num_nodes, 1)  # output layer has one node

        if LOCAL:
            self.weights.append(np.random.rand(*w_dim))
        else:
            self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))

        if not len(self.weights) == (self.num_hidden_layer + 1):
            raise AssertionError("Weights doesn't match number of layers.")

    def train(self, previous_delta_ws=[]):

        def get_prev_delta_w(si, wi):

            if not previous_delta_ws:
                return None
            return previous_delta_ws[si][wi]

        current_delta_ws = []
        for si, sample in enumerate(self.input):
            input = sample
            outputs = []
            for weights in self.weights:
                output = sigmoid(np.dot(input, weights))
                outputs.append(output)
                input = output

            # calculate error term, delta
            output_error = None
            current_delta_w = [None] * len(self.weights)
            for i in reversed(range(len(outputs))):
                if i == 0:
                    input = self.input[si]
                else:
                    input = outputs[i - 1]

                if output_error is None:

                    output_error = sigmoid_derivative(output) * (self.y[si] - output)
                    delta = np.array([input]) * output_error
                    delta_w = self.step_size * delta
                    self.weights[i] += delta_w.T
                    current_delta_w[i] = delta_w.T

                else:
                    error = np.dot(output_error, self.weights[i + 1].T)
                    delta = sigmoid_derivative(outputs[i]) * error  # (T4.4)
                    # delta_w = self.step_size * np.array([[x] for x in input]) * np.array([delta])
                    output_error = delta

                    delta_w = self.step_size * delta * np.array([[x] for x in input])
                    self.weights[i] += delta_w
                    current_delta_w[i] = delta_w

                prev_delta_w = get_prev_delta_w(si, i)
                if prev_delta_w is not None:
                    #print(self.weights[i].shape, prev_delta_w.shape)
                    self.weights[i] += 0.05 * prev_delta_w
            current_delta_ws.append(current_delta_w)

        return current_delta_ws

    def predict(self, samples=None):

        if samples is None:
            samples = self.input
        predictions = []
        for sample in samples:
            input = sample
            for weights in self.weights:
                output = sigmoid(np.dot(input, weights))
                input = output
            predictions.append(output)
        return predictions

    def loss(self, xTests=None):
        if xTests is None:
            xTests = self.input  # calculate loss with yTest
        s = np.sum(np.square(np.array(self.predict(xTests)) - np.array(xTests)))
        return 1 / 2. * s
        #return np.mean(np.square(yTests - self.feedforward()))

# Class definition
class NeuralNetwork2:
    def __init__(self, xTrains, yTrains, num_hidden_layer=2, num_nodes=10,
                 step_size=0.05):
        self.input = np.array(xTrains)  # should be flattened!
        self.y = np.array(yTrains)
        self.num_hidden_layer = num_hidden_layer
        self.num_nodes = num_nodes
        self.step_size = step_size  #0.05
        self.output = np.zeros(self.y.shape)
        self.weights = []
        self.get_initial_weights()
        self.outputs = []

    def get_initial_weights(self):

        for i in range(self.num_hidden_layer):

            if i == 0:
                num_features = len(self.input[0])
                w_dim = (num_features, self.num_nodes)
            else:
                w_dim = (self.num_nodes, self.num_nodes)

            if LOCAL:
                self.weights.append(np.random.rand(*w_dim))
            else:
                self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))

        w_dim = (self.num_nodes, 1)  # output layer has one node

        if LOCAL:
            self.weights.append(np.random.rand(*w_dim))
        else:
            self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))

        if not len(self.weights) == (self.num_hidden_layer + 1):
            raise AssertionError("Weights doesn't match number of layers.")

    def feedforward(self):

        # I tried at lest:(
        # predicted = []
        # for si, sample in enumerate(self.input):
        #     input = sample
        #     outputs = []
        #     for weights in self.weights:
        #         output = sigmoid(np.dot(input, weights))
        #         outputs.append(output)
        #         input = output
        #
        #     predicted.append(output)
        #     # calculate error term, delta
        #     output_error = None
        #     for i in reversed(range(len(outputs))):
        #         if i == 0:
        #             input = self.input[si]
        #         else:
        #             input = outputs[i - 1]
        #
        #         if output_error is None:
        #
        #             output_error = sigmoid_derivative(output) * (self.y[si] - output)
        #             delta = input * output_error
        #             delta_w = self.step_size * delta
        #             self.weights[i] += np.array([[d] for d in delta_w])
        #
        #         else:
        #             error = np.dot(output_error, self.weights[i + 1].T)
        #             delta = sigmoid_derivative(outputs[i]) * error  # (T4.4)
        #             delta_w = self.step_size * np.array([[x] for x in input]) * np.array([delta])
        #             output_error = delta
        #
        #             self.weights[i] += delta_w
        #
        # return np.array(predicted)

        # original
        self.outputs = [self.input]  # initial output from input layer
        outputs = None
        for i, weights_in_layer in enumerate(self.weights):
            if i == 0:
                outputs = sigmoid(np.dot(self.input, weights_in_layer))
            else:
                outputs = sigmoid(np.dot(outputs, weights_in_layer))
            self.outputs.append(outputs)
        return self.outputs[-1]

    def loss(self, yTests=None):
        if yTests is None:
            yTests = self.y  # calculate loss with yTest
        s = np.sum(np.square(np.array(self.feedforward()) - np.array(yTests)))
        return 1 / 2. * s
        #return np.mean(np.square(yTests - self.feedforward()))

    def backward(self):
        # backward propgate through the network
        # self.o_error = y - o  # error in output
        # self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error
        #
        # self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        # self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error
        #
        # self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        # self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

        ## my version - no iteration
        # o_delta = sigmoid_derivative(self.output) * (self.y - self.output)
        #
        # z2_error = np.dot(o_delta, self.weights[1].T)
        # z2_delta = z2_error * sigmoid_derivative(self.outputs[1])
        #
        # delta_w1 = np.dot(self.input.T, z2_delta)
        # delta_w2 = np.dot(self.outputs[1].T, o_delta)
        #
        # self.weights[0] += delta_w1
        # self.weights[1] += delta_w2

        # with iteration
        output_errors = None
        for i in reversed(range(len(self.outputs) - 1)):
            if i == 0:
                input = self.input
            else:
                input = self.outputs[i]

            if output_errors is None:
                output = self.outputs[-1]
                output_errors = sigmoid_derivative(output) * (self.y - output)
                delta_w = self.step_size * np.dot(input.T, output_errors)
                self.weights[i] += delta_w
            else:
                error = np.dot(output_errors, self.weights[i + 1].T)
                delta = sigmoid_derivative(self.outputs[i + 1]) * error  # (T4.4)
                delta_w = self.step_size * np.dot(input.T, delta)
                self.weights[i] += delta_w
                output_errors = delta

    def backprop(self):

        delta_weights = []
        # delta_k = o_k(1 - o_k)(t_k - o_k)
        # error_terms = None
        # for i in reversed(range(len(self.outputs))):
        #     if i == 0:
        #         break
        #     op = self.outputs[i]
        #     ws = self.weights[i - 1]
        #     if error_terms is None:
        #         error_terms = sigmoid_derivative(op) * (self.y - op)
        #         all_deltas = (self.step_size * self.input * error_terms)
        #         delta_weights = np.array([[x] for x in np.sum(all_deltas, axis=1)])
        #     else:
        #         error_terms = sigmoid_derivative(op) * np.dot(ws.T, error_terms)
        #         delta_weights = self.step_size * self.input * error_terms
        #     self.weights[i - 1] += delta_weights

        delta_weights = []
        # delta_k = o_k(1 - o_k)(t_k - o_k)
        propagated_errors = sigmoid_derivative(self.output) * (self.y - self.output)
        for i in reversed(range(len(self.outputs))):
            if i == 0:
                break
            prev_layer_outputs = self.outputs[i - 1]
            d_weights = self.step_size * np.dot(prev_layer_outputs.T, propagated_errors)
            # error_terms = self.step_size * sigmoid_derivative(prev_layer_outputs)*np.dot(self.weights[i-1].T, propagated_errors)
            # d_weights = error_terms * self.input.T
            delta_weights.insert(0, d_weights)
            propagated_errors = np.dot(propagated_errors, self.weights[i - 1].T) * sigmoid_derivative(prev_layer_outputs)

        for i, dw in enumerate(delta_weights):
            self.weights[i] += dw

        ########################################################################
        # dummy case when node 2 and 1 hidden layer.
        # self.layer1, self.layer2 = self.outputs[1], self.outputs[2]
        # self.weights1, self.weights2 = self.weights
        #
        # error_terms_on_output_layer = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        # d_weights2 = np.dot(self.layer1.T, error_terms_on_output_layer)  # np.dot(output from L1.T, error_terms_by_L1)
        # d_weights1 = np.dot(self.input.T,
        #                     np.dot(error_terms_on_output_layer, self.weights2.T) * sigmoid_derivative(self.layer1))
        # # np.dot(
        #
        # self.weights[0] += d_weights1
        # self.weights[1] += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backward()
        #self.backprop()


if __name__ == "__main__":
    # Each row is a training example, each column is a feature  [X1, X2, X3]
    X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    LOCAL = True
    NN = NeuralNetwork(X, y, num_hidden_layer=1, num_nodes=4, step_size=1)
    # for i in range(3000):  # trains the NN 1,000 times
    #     if i % 100 == 0:
    #     #     print("for iteration # " + str(i) + "\n")
    #     #     print("Input : \n" + str(X))
    #         predictions = NN.predict()
    #         print("Actual Output: \n" + str(y))
    #         print("Predicted Output: \n" + str(predictions))
    #
    #         print("Loss: \n" + str(np.mean(np.square(y - predictions))))  # mean sum squared loss
    #         print("My Loss: \n" + str(NN.loss()))
    #     #     print("\n")
    #
    #     NN.train()

    NN = NeuralNetwork2(X, y, num_hidden_layer=1, num_nodes=4, step_size=1)
    for i in range(3000):  # trains the NN 1,000 times
        if i % 100 == 0:
            print("for iteration # " + str(i) + "\n")
            print("Input : \n" + str(X))
            print("Actual Output: \n" + str(y))
            print("Predicted Output: \n" + str(NN.feedforward()))
            print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
            print("My Loss: \n" + str(NN.loss()))
            print("\n")

        NN.train(X, y)
    LOCAL = False

    #####
    from collections import OrderedDict
    import numpy as np
    import os

    from Assignment6.Code import kDataPath
    from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)

    (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw,
                                includeGradients=False,
                                includeRawPixels=False,
                                includeIntensities=True)
    xTrains = np.array([[1] + x for x in xTrains])
    xTests = np.array([[1] + x for x in xTests])
    yTrains = np.array([[y] for y in yTrainRaw])
    yTests = np.array([[y] for y in yTestRaw])



    NN = NeuralNetwork2(xTrains, yTrains, num_hidden_layer=2, num_nodes=4, step_size=0.05)
    for i in range(200):  # trains the NN 1,000 times
        if i % 10 == 0:
            print("for iteration # " + str(i) + "\n")
            #print("Actual Output: \n" + str(y))
            #print("Predicted Output: \n" + str(NN.feedforward())) # mean sum squared loss
            #print("My Loss: \n" + str(NN.loss()))
            predictions = NN.feedforward()
            print("Loss: " + str(np.mean(np.square(yTrains - predictions))))
            #print("\n")

        NN.train(X, y)