from Assignment4.Code import kDataPath
from utils import Assignment4Support as sup
from utils import AddNoise as noise


def get_raw_data():
    """
    :return: (xRaw, yRaw)
    """
    return sup.LoadRawData(kDataPath)


def get_split_data():
    """
    :return:
        xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal
    """
    xRaw, yRaw = get_raw_data()
    return sup.TrainTestSplit(xRaw, yRaw)


def get_xy_train_raw():
    """
    :return: xTrainRaw with noise, yTrainRaw
    """
    (xTrainRawOriginal, yTrainRawOriginal, _, _) = get_split_data()
    return noise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)


def get_xy_test_raw():
    """
    :return: xTestRaw with noise, yTestRaw
    """
    (_, _, xTestRawOriginal, yTestRawOriginal) = get_split_data()
    return noise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)


def get_featurized_xs_ys(numFrequentWords=0,
                         numMutualInformationWords=295,
                         includeHandCraftedFeatures=True):
    """
    :return: xTrain with noise, xTest with noise, yTrain, yTest
    """
    xTrainRaw, yTrainRaw = get_xy_train_raw()
    xTestRaw, yTestRaw = get_xy_test_raw()

    xTrain, xTest = sup.FeaturizeExt(xTrainRaw,
                                     yTrainRaw,
                                     xTestRaw,
                                     numFrequentWords,
                                     numMutualInformationWords,
                                     includeHandCraftedFeatures)
    yTrain = yTrainRaw
    yTest = yTestRaw

    return xTrain, xTest, yTrain, yTest
