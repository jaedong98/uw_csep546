from collections import Counter, namedtuple
import math
import numpy as np

Leaf = namedtuple('Leaf', ['prediction'])

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
        p0 = float(ys_0 + 1 / y_total + 2)
        p1 = float(ys_1 + 1/ y_total + 2)
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
    p0 = float(ys_0/ y_total)
    p1 = float(ys_1/ y_total)
    return -p0 * math.log2(p0) - p1 * math.log2(p1)


def get_information_gains(xTrains, yTrains):

    H = get_entropy_S(yTrains)
    entropies = get_entropy(xTrains, yTrains)
    return list(np.array(H * len(xTrains[0])) - np.array(entropies))


def split(node, min_to_stop=100):

    ((l_xTrains, l_yTrains), (r_xTrains, r_yTrains)) = node['groups']
    del(node['groups'])
    
    if not l_xTrains and r_xTrains:
        node['left'] = None
        node['right'] = get_split(r_xTrains, r_yTrains)
        return

    if l_xTrains and not r_xTrains:
        node['left'] = get_split(l_xTrains, l_yTrains)
        node['right'] = None
        return

    if sum(l_yTrains) == 0 and sum(r_yTrains) == 0:  # all 0
        node['left'] = node['right'] = 0
        return
    
    if all(l_yTrains) and all(r_yTrains):  # all 1
        node['left'] = node['right'] = 1
        return
    
    if len(l_yTrains) < min_to_stop:
        node['left'] = Counter(l_yTrains).most_common(1)[0][0]
    else:
        node['left'] = get_split(l_xTrains, l_yTrains)
        split(node['left'], min_to_stop)

    if len(r_yTrains) < min_to_stop:
        node['right'] = Counter(r_yTrains).most_common(1)[0][0]
    else:
        node['right'] = get_split(r_xTrains, r_yTrains)
        split(node['right'], min_to_stop)


def get_split(xTrains, yTrains):
    """
    Split dataset and create a node based on the feature [i] who has highest information gain.
    Returns:
        an instance of Node
    """
    i_gails = get_information_gains(xTrains, yTrains)
    feature_index = i_gails.index(max(i_gails))
    groups = split_by_feature(feature_index, xTrains, yTrains)
    return {'index': feature_index, 'gain': max(i_gails), 'groups': groups}


def split_by_feature(feature_index, xTrains, yTrains):
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
    threshold = (max(unique_values) - min(unique_values)) / 2

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
    return ((l_xTrains, l_yTrains), (r_xTrains, r_yTrains))


# Build a decision tree
def build_tree(xTrains, yTrains, min_to_stop=100):
    root = get_split(xTrains, yTrains)
    split(root, min_to_stop)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('{} Feature {}: \n'.format(str(depth * ' '), node['index']))
        if 'left' in node:
            print_tree(node['left'], depth+1)
        if 'right' in node:
            print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % (depth * ' ', node))


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
    print(split_by_feature(0, xTrains, yTrains))
    print(split_by_feature(1, xTrains, yTrains))
    print(split_by_feature(2, xTrains, yTrains))

    root = get_split(xTrains, yTrains)
    print(root)

    tree = build_tree(xTrains, yTrains, 2)
    print(tree)
    print_tree(tree)
