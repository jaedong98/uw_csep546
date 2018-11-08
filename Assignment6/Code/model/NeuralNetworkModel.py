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
    def __init__(self, xTrains, yTrains, num_nodes=2):
        self.input = xTrains
        self.weights1 = np.random.rand(self.input.shape[1], num_nodes)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = yTrains
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))


class Neuron(object):

    def __init__(self, num_pixels=144):

        ini_weights = [random.uniform(-0.05, 0.05) for _ in range(num_pixels)]
        self.weights = [1] + ini_weights

    def predict(self, example):
        """Calculate O_u"""
        # Mitchel page 96. Feature 4.6
        example_ext = [1] + example
        net = np.dot(example_ext, self.weights)
        o = 1. / (1 + np.exp(-net))
        return o


class HiddenLayer(object):

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [Neuron() for _ in range(num_neurons)]

    def predict(self, example):
        return [n.predict(example) for n in self.neurons]


