import utils.Assignment5Support as sup
from Assignment5.Code import kDataPath

(xRaw, yRaw) = sup.LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = sup.TrainTestSplit(xRaw, yRaw, percentTest = .25)
print("Calculating features...")
(xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yTest = yTestRaw

