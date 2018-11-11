import os
import random

from utils.cv_features import get_y_gradient_features, get_x_gradient_features, get_y_gradient_histogram_features, \
    get_x_gradient_histogram_features


def LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True, shuffle=True):
    xRaw = []
    yRaw = []

    if includeLeftEye:
        closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openLeftEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if includeRightEye:
        closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openRightEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if shuffle:
        random.seed(1000)

        index = [i for i in range(len(xRaw))]
        random.shuffle(index)

        xOrig = xRaw
        xRaw = []

        yOrig = yRaw
        yRaw = []

        for i in index:
            xRaw.append(xOrig[i])
            yRaw.append(yOrig[i])

    return (xRaw, yRaw)


def TrainTestSplit(x, y, percentTest=.25):
    if(len(x) != len(y)):
        raise UserWarning(
            "Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning(
            "Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)


from PIL import Image

def Convolution3x3(image, filter):
    # check that the filter is formated correctly
    if not (len(filter) == 3 and len(filter[0]) == 3 and len(filter[1]) == 3 and len(filter[2]) == 3):
        raise UserWarning("Filter is not formatted correctly, should be [[x,x,x], [x,x,x], [x,x,x]]")

    xSize = image.size[0]
    ySize = image.size[1]
    pixels = image.load()

    answer = []
    for x in range(xSize):
        answer.append([ 0 for y in range(ySize) ])

    # skip the edges
    for x in range(1, xSize - 1):
        for y in range(1, ySize - 1):
            value = 0

            for filterX in range(len(filter)):
                for filterY in range(len(filter)):
                    imageX = x + (filterX - 1)
                    imageY = y + (filterY - 1)

                    value += pixels[imageX, imageY] * filter[filterX][filterY]

            answer[x][y] = value

    return answer


def Featurize(xTrainRaw,
              xTestRaw,
              includeGradients=False,
              includeRawPixels=False,
              includeIntensities=True,
              grid_y_gradients=False,
              grid_x_gradients=False,
              hist_y_gradients=False,
              hist_x_gradients=False
              ):
    """
    from PIL import Image

    # if path is in right format…
    image = Image.open(<path>)
    pixels = image.load()

    intensity = pixels[1, 1] / 255.0

    :param xTrainRaw:
    :param xTestRaw:
    :param includeGradients:
    :param includeRawPixels:
    :param includeIntensities:
    :return:
    """
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for sample in xTrainRaw:
        features = []

        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])


        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        if grid_y_gradients:
            features.extend(get_y_gradient_features(image, grid_dim=(3, 3)))

        if grid_x_gradients:
            features.extend(get_x_gradient_features(image, grid_dim=(3, 3)))

        if hist_y_gradients:
            features.extend(get_y_gradient_histogram_features(image))

        if hist_x_gradients:
            features.extend(get_x_gradient_histogram_features(image))

        xTrain.append(features)

    print("Featurized Training Set.")
    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for sample in xTestRaw:
        features = []
        
        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            yEdges = Convolution3x3(image, [[1, 0, -1],[2,0,-2],[1,0,-1]])
            sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            count = sum([len(row) for row in yEdges])

            features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            count = sum([len(row[8:16]) for row in yEdges])

            features.append(sumGradient / count)

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])

        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        if grid_y_gradients:
            features.extend(get_y_gradient_features(image, grid_dim=(3, 3)))

        if grid_x_gradients:
            features.extend(get_x_gradient_features(image, grid_dim=(3, 3)))

        if hist_y_gradients:
            features.extend(get_y_gradient_histogram_features(image))

        if hist_x_gradients:
            features.extend(get_x_gradient_histogram_features(image))

        xTest.append(features)

    print("Featurized Test Set.")
    return (xTrain, xTest)


import PIL
from PIL import Image

def VisualizeWeights(weightArray, outputPath, resize=()):
    size = 12
    # note the extra weight for the bias is where the +1 comes from, just ignore it
    if len(weightArray) != (size*size) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (size*size) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("L", (size,size))

    pixels = image.load()

    for x in range(size):
        for y in range(size):
            pixels[x,y] = int(abs(weightArray[(x*size) + y]) * 255)

    if resize is not None:
        image = image.resize(resize)
    image.save(outputPath)