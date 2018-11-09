import os
from Assignment6.Code import kDataPath, report_path
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize


def run(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
        num_hidden_layers=[1, 2],
        num_nodes_per_hideen_layer=[2, 5, 10, 15, 20],
        iterations=200,
        step_size=0.05):
    (xTrain, xTest) = Featurize(xTrainRaw, xTestRaw,
                                includeGradients=False,
                                includeRawPixels=False,
                                includeIntensities=True)
    yTrain = yTrainRaw
    yTest = yTestRaw



if __name__ == "__main__":
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)


