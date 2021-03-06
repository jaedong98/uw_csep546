from collections import OrderedDict
import numpy as np
import os

from Assignment6.Code import kDataPath, report_path
from model.NeuralNetworkModel import NeuralNetwork
from utils.Assignment4Support import draw_loss_comparisions
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize
from utils.EvaluationsStub import Evaluation


prob2_report_path = os.path.join(report_path, 'prob2')


def run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
        num_hidden_layer=2,
        num_nodes=15,
        iterations=200,
        step_size=0.05,
        norm_factor=255.,
        includeIntensities=True,
        decrease_size_by=2,
        momentum=0.0):
    (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw,
                                includeGradients=False,
                                includeRawPixels=False,
                                includeIntensities=includeIntensities,
                                norm_factor=norm_factor,
                                decrease_size_by=decrease_size_by)
    xTrains = np.array([[1] + x for x in xTrains])
    xTests = np.array([[1] + x for x in xTests])
    yTrains = np.array([[y] for y in yTrainRaw])
    yTests = np.array([[y] for y in yTestRaw])
    legends = []
    training_loss_data = OrderedDict()
    test_loss_data = OrderedDict()
    test_accuracy_data = OrderedDict()

    case = (num_hidden_layer, num_nodes)
    training_loss_data[case] = []
    test_loss_data[case] = []
    test_accuracy_data[case] = []
    NN = NeuralNetwork(xTrains, yTrains,
                       num_hidden_layer=num_hidden_layer,
                       num_nodes=num_nodes,
                       step_size=step_size,
                       momentum=momentum)
    predictions = np.zeros(yTrains.shape)
    previous_delta_ws = []
    best_accuracy = -1
    best_eval = None
    best_iteration = None
    for i in range(iterations):

        # outputs = NN.feedforward()
        # loss = np.mean(np.square(yTrains - outputs))
        # training_loss_data[case].append((i, loss))
        # if i % 50 == 0:  # mean sum squared loss
        #     print("Case: " + str(case) + " Loss: \n" + str(loss))
        #     print("\n")
        # NN.train(xTrains, yTrains)

        predictions = NN.predict()
        loss = np.sum(np.square(yTrains - predictions)) / 2.
        training_loss_data[case].append((i, loss))

        predictions = NN.predict(xTests)
        # test_loss = np.mean(np.square(yTests - predictions))
        test_loss = np.sum(np.square(yTests - predictions)) / 2.
        test_ev = Evaluation([x[0] for x in yTests],[1 if x[0] >= 0.5 else 0 for x in predictions])
        test_accuracy_data[case].append((i, test_ev.accuracy))
        test_loss_data[case].append((i, test_loss))
        if test_ev.accuracy > best_accuracy:
            best_eval = test_ev
            best_accuracy = test_ev.accuracy
            best_iteration = i
        if i % 10 == 0:
            print("Loss: " + str(loss))  # mean sum squared loss
            print("Test Loss: " + str(test_loss))
            print("Accuracy: {}".format(test_ev.accuracy))

        previous_delta_ws = NN.train(previous_delta_ws)

    test_ev = Evaluation([x[0] for x in yTests],
                         [1 if x[0] >= 0.5 else 0 for x in predictions])
    print("Accuracy after iteration: {}".format(test_ev.accuracy))

    best_accuracy_md = os.path.join(prob2_report_path,
                                    'prob2_accuracy_{}_{}_{}_w_momentum_{}.md'
                                    .format(norm_factor, step_size,
                                            decrease_size_by,
                                            momentum))
    with open(best_accuracy_md, 'w') as f:
        f.write("Best Accuracy at {} / {} iterations"
                .format(best_iteration, iterations))
        f.write('\n')
        f.write(str(best_eval))
        f.write('\n')
        f.write(str(test_ev))

    training_loss_fname = os.path.join(prob2_report_path,
                                       "prob2_training_loss_"
                                       "case_{}_{}_{}_ss{}_{}_w_momentum_{}.png"
                                       .format(num_hidden_layer,
                                               num_nodes,
                                               norm_factor,
                                               step_size,
                                               decrease_size_by,
                                               momentum))
    case_legend = ['{} hidden layer with {} nodes w/ momentum {}'
                       .format(num_hidden_layer, num_nodes, momentum)]
    draw_loss_comparisions([training_loss_data[case]], "Iterations", "Loss", "Training Set",
                           training_loss_fname, case_legend,
                           data_pt='-',
                           title_y=1)

    test_loss_fname = os.path.join(prob2_report_path,
                                   "prob2_test_loss_"
                                   "case_{}_{}_{}_ss{}_{}_w_momentum_{}.png"
                                   .format(num_hidden_layer,
                                           num_nodes,
                                           norm_factor,
                                           step_size,
                                           decrease_size_by,
                                           momentum))
    draw_loss_comparisions([test_loss_data[case]], "Iterations", "Loss", "Test Set",
                           test_loss_fname, case_legend,
                           data_pt='-',
                           title_y=1)


if __name__ == "__main__":
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)

    #paramters with best accuracy
    for momentum in [0.0]: #, 0.1, 0.25, 0.5, 0.75, 1.0]:
        for norm_factor in [255.0]:
            run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                num_hidden_layer=2,
                num_nodes=15,
                iterations=200,
                step_size=.08,
                norm_factor=norm_factor,
                decrease_size_by=2,
                momentum=momentum)

