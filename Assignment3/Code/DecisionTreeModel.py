import math
import numpy as np


class DecisionTree():

    def __init__(self, minToSplit=100):
        self.minToSplit = minToSplit

    def fit(self, x, y):
        pass

    def predict(self, xTest):
        pass

    def visualize(self):
        pass


def get_entropy(xTrains, yTrains):
    """Calculates entropies for all features in S(xTrains)"""
    if all(yTrains):  # all ys are 1
        return [0] * len(xTrains[0])

    if not any(yTrains):  # all ys are 0
        return [0] * len(xTrains[0])

    entropies = list()
    for i in range(len(xTrains[0])):
        ys_0 = 0
        ys_1 = 0
        for xTrain, yTrain in zip(xTrains, yTrains):
            if xTrain[i] == 1:
                if yTrain == 1:
                    ys_1 += 1
                else:
                    ys_0 += 1
        if ys_0 == ys_1:
            # entropy is 1 when the collection contains an equal number of positive and negative examples.
            entropies.append(1)
            continue
        y_total = ys_0 + ys_1
        p0 = float(ys_0 / y_total)
        p1 = float(ys_1 / y_total)
        entropies.append(-p0 * math.log2(p0) - p1 * math.log2(p1))

    return entropies


def get_entropy_S(yTrains):
    """Calculate entropy of yTrains. In tree, yTrains will be the splitted yTrains."""
    if all(yTrains):  # all ys are 1
        return 0

    if not any(yTrains):  # all ys are 0
        return 0

    ys_0 = 0
    ys_1 = 0
    for y in yTrains:
        if y == 1:
            ys_1 += 1
        else:
            ys_0 += 1
    if ys_0 == ys_1:
        return 1

    y_total = ys_0 + ys_1
    p0 = float(ys_0 / y_total)
    p1 = float(ys_1 / y_total)
    return -p0 * math.log2(p0) - p1 * math.log2(p1)


def get_information_gains(xTrains, yTrains):

    H = get_entropy_S(yTrains)
    entropies = get_entropy(xTrains, yTrains)
    return list(np.array(H * len(xTrains[0])) - np.array(entropies))


def split(feature_index, xTrains, yTrains):
    """
        Calculates the threshold based on the feature values at i and split data 
        into two groups.
    """
    if not len(xTrains) == len(yTrains):
        raise ValueError("Unmatched xTrains({}) and yTrains({})".format(
            len(xTrains), len(yTrains)))

    if feature_index > len(xTrains):
        raise IndexError(
            "Feature index({}) is outside of xTrains".format(feature_index))

    # gather all values of feature xTrain[i]
    values_by_features = [x for x in zip(*xTrains)]
    unique_values = list(set(values_by_features[feature_index]))
    threshold = sum(unique_values) / len(unique_values)

    l_xTrains, l_yTrains = [], []
    r_xTrains, r_yTrains = [], []
    for xTrain, yTrain in zip(xTrains, yTrains):
        if xTrain[feature_index] < threshold:
            l_xTrains.append(xTrain)
            l_yTrains.append(yTrain)
        else:
            r_xTrains.append(xTrain)
            r_yTrains.append(yTrain)

    print("{} samples are splitted into L({}), R({})".format(
        len(xTrains), len(l_xTrains), len(r_xTrains)))
    return l_xTrains, l_yTrains, r_xTrains, r_yTrains


if __name__ == '__main__':
    xTrains = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
               [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    yTrains = [1, 0, 1, 1, 0, 1, 0, 1]
    # [1, -0.8112781244591328, -0.8112781244591328]
    print(get_entropy(xTrains, yTrains))
    print(get_entropy_S(yTrains))
    print("Information Gains: {}".format(
        get_information_gains(xTrains, yTrains)))

    yTrains = [0, 0, 0, 0, 1, 1, 1, 1]
    print(get_entropy_S(yTrains))

    yTrains = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print(get_entropy_S(yTrains))

    yTrains = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(get_entropy(xTrains, yTrains))  # [0, 0, 0]
    print(get_entropy_S(yTrains))

    yTrains = [1, 1, 1, 1, 1, 1, 1, 1]
    print(get_entropy(xTrains, yTrains))  # [0, 0, 0]
    print(get_entropy_S(yTrains))

    yTrains = [0, 1, 0, 1, 0, 1, 0, 1]
    print(split(0, xTrains, yTrains))
    print(split(1, xTrains, yTrains))
    print(split(2, xTrains, yTrains))
