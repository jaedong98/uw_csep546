import collections


def LoadRawData(path):
    f = open(path, 'r')

    lines = f.readlines()

    kNumberExamplesExpected = 5574

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (
            kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)


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


def FeaturizeTraining(xTrainRaw, feature_selection_functions):
    """
    xTrainRaw: a list of spam messages.
    feature_section_functions: a list of f(msg)s
    """
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        for fsf in feature_selection_functions:
            features.append(fsf(x))

        xTrain.append(features)

    return xTrain


def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])


def most_frequent_features(xTrain, N=1):

    count = collections.Counter()
    for x in xTrain:
        words = x.split(' ')
        for word in words:
            count[word] += 1

    return count.most_common(N)


def is_longger(msg, threshold=40):
    if len(msg) > threshold:
        return 1
    return 0


def has_number(msg):
    if any(i.isdigit() for i in msg):
        return 1
    return 0


def contain_word(msg, word):
    if word in msg:
        return 1
    return 0


def contain_call(msg):
    return contain_word(msg, 'call')


def contain_to(msg):
    return contain_word(msg, 'to')


def contain_your(msg):
    return contain_word(msg, 'your')
