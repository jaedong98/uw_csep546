#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 2, 2018
@author: Madhuri Suthar, PhD Candidate in Electrical and Computer Engineering, UCLA
"""

# Imports
import numpy as np
np.random.seed(100)
# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)


# Define useful functions

# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y, num_nodes=2):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], num_nodes)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(num_nodes, 1)  # weights
        #self.weights3 = np.random.rand(num_nodes, 1)  # weights for output
        self.y = y
        self.output = np.zeros(y.shape)  # y_hat vector

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        #self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer2

    def backprop(self):

        error_terms_on_output_layer = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, error_terms_on_output_layer)  # np.dot(output from L1.T, error_terms_by_L1)
        d_weights1 = np.dot(self.input.T, np.dot(error_terms_on_output_layer, self.weights2.T) * sigmoid_derivative(self.layer1))
        # np.dot(

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")

    NN.train(X, y)