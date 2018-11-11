from collections import OrderedDict
import inspect
import numpy as np
import os
from Assignment6.Code import kDataPath, report_path
from model.NeuralNetworkModel import NeuralNetwork
from model.RandomForestsModel import RandomForestModel
from utils.Assignment4Support import draw_accuracies
from utils.Assignment5Support import Featurize, TrainTestSplit, LoadRawData
from utils.EvaluationsStub import Evaluation

prob2_report_path = os.path.join(report_path, 'prob2')


def compare_roc_curves_nn_rf(xTrainRaw, yTrainRaw,
                             xTestRaw, yTestRaw,
                             config, thresholds,
                             num_hidden_layer=2,
                             num_nodes=15,
                             step_size=0.08,
                             iterations=200):

    graphs = []
    legends = []


    (xTrain, xTest) = Featurize(xTrainRaw, xTestRaw,
                                includeGradients=False,
                                includeRawPixels=False,
                                includeIntensities=True)
    yTrain = yTrainRaw
    yTest = yTestRaw

    print("Running RandomForest")
    legends.append("RandomForest")
    model = RandomForestModel(numTrees=config['num_trees'],
                              bagging_w_replacement=config['bagging_w_replacement'])
    model.fit(xTrain, yTrain, min_to_split=config['min_to_split'])
    yTestPredicted_prob = model.predict_probabilities(xTest)

    cont_length_fpr_fnr = []
    for threshold in thresholds:
        yTestPredicted = [int(p >= threshold) for p in yTestPredicted_prob]
        # featured = [int(y < threshold) for y in yTestPredicted]
        ev = Evaluation(yTest, yTestPredicted)
        # if threshold == 0:
        #    assert (ev.fpr == 0.0), ev
        print(ev)
        cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
    graphs.append(cont_length_fpr_fnr)

    #
    print("NeuralNetwork")
    legends.append("NeuralNetwork")
    xTrains = np.array([[1] + x for x in xTrain])
    xTests = np.array([[1] + x for x in xTest])
    yTrains = np.array([[y] for y in yTrainRaw])
    yTests = np.array([[y] for y in yTestRaw])

    case = (num_hidden_layer, num_nodes)
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

        predictions = NN.predict(xTests)
        # test_loss = np.mean(np.square(yTests - predictions))
        test_loss = np.sum(np.square(yTests - predictions)) / 2.
        test_ev = Evaluation([x[0] for x in yTests], [1 if x[0] >= 0.5 else 0 for x in predictions])
        if i % 10 == 0:
            print("Loss: " + str(loss))  # mean sum squared loss
            print("Test Loss: " + str(test_loss))
            print("Accuracy: {}".format(test_ev.accuracy))

        NN.train()

    yTestPredicted_prob = [x[0] for x in predictions]
    cont_length_fpr_fnr = []
    for threshold in thresholds:
        yTestPredicted = [int(p >= threshold) for p in yTestPredicted_prob]
        # featured = [int(y < threshold) for y in yTestPredicted]
        ev = Evaluation(yTest, yTestPredicted)
        # if threshold == 0:
        #    assert (ev.fpr == 0.0), ev
        print(ev)
        cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
    graphs.append(cont_length_fpr_fnr)


    # plotting
    start = thresholds[0]
    end = thresholds[-1]
    step = thresholds[1] - thresholds[0]
    fname = 'prob2{}_{}_{}_{}.png'.format(inspect.stack()[0][3], start, end, step)
    img_fname = os.path.join(prob2_report_path, fname)
    draw_accuracies(graphs,
                    'False Positive Rate', 'False Negative Rate',
                    'ROC Curve Comparision',
                    img_fname,
                    legends=legends,
                    invert_yaxis=True,
                    data_pt='-',
                    title_y=1.05)


if __name__ == '__main__':
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)

    config = {'min_to_split': 2,
              'bagging_w_replacement': True,
              'num_trees': 40,
              'feature_restriction': 0}
    start = 0
    end = 1
    N = 100
    thresholds = [x / N for x in range(N + 1)]
    compare_roc_curves_nn_rf(xTrainRaw, yTrainRaw,
                             xTestRaw, yTestRaw,
                             config, thresholds,
                             num_hidden_layer=2,
                             num_nodes=15,
                             step_size=0.08,
                             iterations=200)
