from collections import OrderedDict
import numpy as np
import os

from Assignment6.Code import kDataPath, report_path
from model.NeuralNetworkModel import NeuralNetwork
from utils.Assignment4Support import draw_loss_comparisions
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize


def run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
        num_hidden_layers=[1, 2],
        num_nodes_per_hideen_layer=[2, 5, 10, 15, 20],
        iterations=200,
        step_size=0.05):
    (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw,
                                includeGradients=False,
                                includeRawPixels=False,
                                includeIntensities=True)
    xTrains = np.array([[1] + x for x in xTrains])
    xTests = np.array([[1] + x for x in xTests])
    yTrains = np.array([[y] for y in yTrainRaw])
    yTests = np.array([[y] for y in yTestRaw])
    legends = []
    training_loss_data = OrderedDict()
    test_loss_data = OrderedDict()
    #xTrains = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
    #yTrains = np.array(([0], [1], [1], [0]), dtype=float)

    # NN = NeuralNetwork(xTrains, yTrains, num_hidden_layer=1, num_nodes=4, step_size=1)
    # for i in range(3000):  # trains the NN 1,000 times
    #     if i % 100 == 0:
    #         print("for iteration # " + str(i) + "\n")
    #         print("Input : \n" + str(xTrains))
    #         print("Actual Output: \n" + str(yTrains))
    #         print("Predicted Output: \n" + str(NN.feedforward()))
    #         print("Loss: \n" + str(np.mean(np.square(yTrains - NN.feedforward()))))  # mean sum squared loss
    #         print("My Loss: \n" + str(NN.loss()))
    #         print("\n")
    #
    #     NN.train(xTrains, yTrains)
    # return
    for num_hidden_layer in num_hidden_layers:
        for num_nodes in num_nodes_per_hideen_layer:
            legends.append('{}_hidden(s)_{}_nodes'.format(num_hidden_layer, num_nodes))
            case = (num_hidden_layer, num_nodes)
            training_loss_data[case] = []
            test_loss_data[case] = []
            NN = NeuralNetwork(xTrains, yTrains,
                               num_hidden_layer=num_hidden_layer,
                               num_nodes=num_nodes,
                               step_size=step_size)

            for i in range(iterations):

                # outputs = NN.feedforward()
                # loss = np.mean(np.square(yTrains - outputs))
                # training_loss_data[case].append((i, loss))
                # if i % 50 == 0:  # mean sum squared loss
                #     print("Case: " + str(case) + " Loss: \n" + str(loss))
                #     print("\n")
                # NN.train(xTrains, yTrains)

                predictions = NN.predict()
                loss = np.mean(np.square(yTrains - predictions))
                training_loss_data[case].append((i, loss))

                predictions = NN.predict(xTests)
                test_loss = np.mean(np.square(yTests - predictions))
                test_loss_data[case].append((i, test_loss))
                if i % 10 == 0:
                    print("Loss: " + str(loss))  # mean sum squared loss
                    print("Test Loss: " + str(test_loss))
                NN.train()

            training_loss_fname = os.path.join(report_path,
                                               "prob1_training_loss_"
                                               "case_{}_{}.png"
                                               .format(num_hidden_layer,
                                                       num_nodes))
            case_legend = ['{} hidden layer with {} nodes'.format(num_hidden_layer, num_nodes)]
            draw_loss_comparisions([training_loss_data[case]], "Iterations", "Loss", "Training Set",
                                   training_loss_fname, case_legend,
                                   data_pt='-',
                                   title_y=1)

            test_loss_fname = os.path.join(report_path,
                                               "prob1_test_loss_"
                                               "case_{}_{}.png"
                                               .format(num_hidden_layer,
                                                       num_nodes))
            draw_loss_comparisions([test_loss_data[case]], "Iterations", "Loss", "Test Set",
                                   test_loss_fname, case_legend,
                                   data_pt='-',
                                   title_y=1)


    training_loss_fname = os.path.join(report_path, "prob1_training_loss_{}_{}.png".format(max(num_hidden_layers), max(num_nodes_per_hideen_layer)))
    draw_loss_comparisions(training_loss_data.values(), "Iterations", "Loss", "Training Set",
                           training_loss_fname, legends,
                           data_pt='-',
                           title_y=1)

    test_loss_fname = os.path.join(report_path, "prob1_test_loss_{}_{}.png".format(max(num_hidden_layers), max(num_nodes_per_hideen_layer)))
    draw_loss_comparisions(test_loss_data.values(), "Iterations", "Loss", "Test Set",
                           test_loss_fname, legends,
                           data_pt='-',
                           title_y=1)


if __name__ == "__main__":
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)
    run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
        num_hidden_layers=[1, 2],
        num_nodes_per_hideen_layer=[2, 5, 10, 15, 20],
        iterations=200,
        step_size=.05)

