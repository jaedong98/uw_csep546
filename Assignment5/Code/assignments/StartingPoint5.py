## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

import utils.Assignment5Support as sup

## NOTE update this with your equivalent code..
from Assignment5.Code import kDataPath
from utils.Assignment4Support import calculate_bounds
from utils.EvaluationsStub import Evaluation

(xRaw, yRaw) = sup.LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = sup.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xTest) = sup.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yTest = yTestRaw


######
import model.MostCommonModel as mc
model = mc.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)
ev = Evaluation(yTest, yTestPredicted)
upper, lower = calculate_bounds(ev.accuracy, zn=1.96, N=len(yTest))
print("Most Common Accuracy:", ev.accuracy, upper, lower)

######
import model.DecisionTreeModel as dt
model = dt.DecisionTreeModel()
model.fit(xTrain, yTrain, min_to_stop=50)
yTestPredicted = model.predict(xTest)
ev = Evaluation(yTest, yTestPredicted)
upper, lower = calculate_bounds(ev.accuracy, zn=1.96, N=len(yTest))
print("Decision Tree Accuracy:", ev.accuracy, upper, lower)

import model.GeoffDecisionTreeModel as gdt
model = gdt.DecisionTreeModel()
model.fit(xTrain, yTrain, minToSplit=50)
yTestPredicted = model.predict(xTest)
ev = Evaluation(yTest, yTestPredicted)
upper, lower = calculate_bounds(ev.accuracy, zn=1.96, N=len(yTest))
print("Geof's Decision Tree Accuracy:", ev.accuracy, upper, lower)


##### for visualizing in 2d
#for i in range(500):
#    print("%f, %f, %d" % (xTrain[i][0], xTrain[i][1], yTrain[i]))

##### sample image debugging output

import PIL
from PIL import Image

i = Image.open(xTrainRaw[1])
#i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")

print(i.format, i.size)

# Sobel operator
xEdges = sup.Convolution3x3(i, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
yEdges = sup.Convolution3x3(i, [[1, 0, -1], [2, 0, -2], [1, 0, -1]])

pixels = i.load()

for x in range(i.size[0]):
    for y in range(i.size[1]):
        pixels[x,y] = abs(xEdges[x][y])

#i.save("c:\\Users\\ghult\\Desktop\\testEdgesY.jpg")