from collections import OrderedDict
import numpy as np
import os

from Assignment6.Code import kDataPath, report_path
from model.NeuralNetworkModel import NeuralNetwork
from utils.Assignment4Support import draw_loss_comparisions
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize, VisualizeWeights
from utils.EvaluationsStub import Evaluation


def run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
        num_hidden_layers=[1, 2],
        num_nodes_per_hideen_layer=[2, 5, 10, 15, 20],
        iterations=200,
        step_size=0.05,
        weights_on_image=False):
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
    test_accuracy_data = OrderedDict()

    for num_hidden_layer in num_hidden_layers:
        for num_nodes in num_nodes_per_hideen_layer:
            legends.append('{}_hidden(s)_{}_nodes'.format(num_hidden_layer, num_nodes))
            case = (num_hidden_layer, num_nodes)
            training_loss_data[case] = []
            test_loss_data[case] = []
            test_accuracy_data[case] = []
            NN = NeuralNetwork(xTrains, yTrains,
                               num_hidden_layer=num_hidden_layer,
                               num_nodes=num_nodes,
                               step_size=step_size)
            predictions = np.zeros(yTrains.shape)
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
                # test_loss = np.mean(np.square(yTests - predictions))
                test_loss = np.sum(np.square(yTests - predictions)) / 2.
                test_ev = Evaluation([x[0] for x in yTests],[1 if x[0] >= 0.5 else 0 for x in predictions])
                test_accuracy_data[case].append((i, test_ev.accuracy))
                test_loss_data[case].append((i, test_loss))
                if i % 10 == 0:
                    print("Loss: " + str(loss))  # mean sum squared loss
                    print("Test Loss: " + str(test_loss))
                    print("Accuracy: {}".format(test_ev.accuracy))

                NN.train()
            test_ev = Evaluation([x[0] for x in yTests],
                                 [1 if x[0] >= 0.5 else 0 for x in predictions])
            print("Accuracy after iteration: {}".format(test_ev.accuracy))

            if weights_on_image:
                for i, w in enumerate(NN.weights):
                    img_path = os.path.join(report_path,
                                            "weights_node{}_of_{}_in_{}_layer"
                                            ".jpg".format(i, num_nodes,
                                                          num_hidden_layer))
                    VisualizeWeights(NN.weights[0][:, 1], img_path, resize=(288, 288))

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

    test_accuracy_fname = os.path.join(report_path, "prob1_test_accuracy_{}_{}.png".format(max(num_hidden_layers), max(
        num_nodes_per_hideen_layer)))
    draw_loss_comparisions(test_accuracy_data.values(), "Iterations", "Accuracy", "Accuracies on Test Set",
                           test_accuracy_fname, legends,
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
        step_size=.05,
        weights_on_image=True)

