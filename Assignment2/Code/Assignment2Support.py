import collections
import matplotlib
import matplotlib.pyplot as plt


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

def exclude(msg):
    return 1

def draw_single_plot(tuples, xlabel, ylabel, title, img_fname):
    t, s = zip(*tuples)
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def draw_weights(iter_cnts, weights, xlabel, ylabel, title, img_fname):
    """
    iter_cnts: a list of iteration counts, [1000, 2000, ...]
    weights: a list of weights per iteration, [(0, 0,...), (0.1, 0.2,...),...]
    """
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for ws in zip(*weights):
        ax.plot(iter_cnts, ws)

    ax.legend(('w0', 'w1', 'w2', 'w3', 'w4', 'w5'))
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))


def draw_accuracies(accuracies, xlabel, ylabel, title, img_fname, legends):
    """
    accuracies: a list of list of (iter_cnt, accurarcy)s
    legends: a tuple/list of legends for each graphs
    """
    if not len(accuracies) == len(legends):
        raise ValueError("Missing legends? {} vs {}".format(
            len(accuracies), len(legends)))

    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    for accus in accuracies:
        xs, ys = zip(*accus)
        ax.plot(xs, ys)

    ax.legend(legends)
    ax.grid()
    fig.savefig(img_fname)
    print("Saved/Updated image {}".format(img_fname))

def accuracy_table(accuracies, features, w=20):

    table = '|{}|{}|'.format('Leave-out-Features'.center(w), 'Accuracy'.center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|'
    for feature, accu in zip(features, accuracies):
        table += '\n|{}|{}|'.format('{}'.format(feature).center(w), '{}'.format(accu[-1][-1]).center(w))
    return table

def selected_features_table(features, headers, w=20):
    """
    features: a list of two element tuples, [('to', 1586), ('you', 1174),...]
    headers: a list of two headers, ["Feature", "Frequency"]
    """
    table = '|{}|{}|'.format(headers[0].center(w), headers[1].center(w))
    table += '\n|' + '-' * w
    table += '|' + '-' * w
    table += '|'
    for feature, num in features:
        table += '\n|{}|{}|'.format('{}'.format(feature).center(w), '{}'.format(num).center(w))
    return table