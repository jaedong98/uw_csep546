"""
References:
    https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
    https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

"""
import numpy as np
np.random.seed(100)


# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, xTrains, yTrains, num_hidden_layers=2, num_nodes=10,
                 step_size=0.05):
        self.input = np.array(xTrains)  # should be flattened!
        self.y = np.array(yTrains)
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        self.step_size = 1  #0.05
        self.output = np.zeros(y.shape)
        self.weights = []
        self.get_initial_weights()
        self.layers = []

    def get_initial_weights(self):

        for i in range(self.num_hidden_layers):

            if i == 0:
                num_features = len(self.input[0])
                w_dim = (num_features, self.num_nodes)
            else:
                w_dim = (self.num_nodes, self.num_nodes)
            #self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))
            self.weights.append(np.random.rand(*w_dim))

        w_dim = (self.num_nodes, 1)  # output layer has one node
        #self.weights.append(np.random.uniform(-0.05, 0.05, w_dim))
        self.weights.append(np.random.rand(*w_dim))

        if not len(self.weights) == (self.num_hidden_layers + 1):
            raise AssertionError("Weights doesn't match number of layers.")

    def feedforward(self):

        self.layers = [self.input]  # initial output from input layer
        outputs = None
        for weights_in_layer in self.weights:
            if outputs is None:
                outputs = sigmoid(np.dot(self.input, weights_in_layer))
                self.layers.append(outputs)
            else:
                outputs = sigmoid(np.dot(outputs, weights_in_layer))
                self.layers.append(outputs)
        return outputs

    def loss(self, yTests=None):
        if yTests is None:
            yTests = self.y  # calculate loss with yTest
        s = np.sum(np.square(np.array(self.feedforward()) - np.array(yTests)))
        return 1 / 2. * s

    def backprop(self):

        delta_weights = []
        propagated_errors = (self.y - self.output) * sigmoid_derivative(self.output)
        for i in reversed(range(len(self.layers))):
            if i == 0:
                break
            prev_layer_outputs = self.layers[i - 1]
            d_weights = self.step_size * np.dot(prev_layer_outputs.T, propagated_errors)
            delta_weights.insert(0, d_weights)
            propagated_errors = np.dot(propagated_errors, self.weights[i - 1].T) * sigmoid_derivative(prev_layer_outputs)

        for i, dw in enumerate(delta_weights):
            self.weights[i] += dw

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


if __name__ == "__main__":
    # Each row is a training example, each column is a feature  [X1, X2, X3]
    X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    NN = NeuralNetwork(X, y, num_nodes=4)
    for i in range(3000):  # trains the NN 1,000 times
        if i % 100 == 0:
            print("for iteration # " + str(i) + "\n")
            print("Input : \n" + str(X))
            print("Actual Output: \n" + str(y))
            print("Predicted Output: \n" + str(NN.feedforward()))
            print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
            print("\n")

        NN.train(X, y)