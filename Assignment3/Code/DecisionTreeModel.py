from collections import Counter, namedtuple
import math


class DecisionTreeModel(object):

    def __init__(self):
        self.tree = None

    def fit(self, x, y, min_to_stop=100):
        self.tree = build_tree(x, y, min_to_stop)

    def predict(self, xTest, threshold=None):

        predictions = []
        for example in xTest:
            if threshold:
                predictions.append(predict_w_threshold(self.tree, example, threshold))
            else:
                predictions.append(predict(self.tree, example))

        return predictions

    def visualize(self, file_obj=None):
        print_tree(self.tree)
        if file_obj:
            write_tree(self.tree, file_obj)



def get_entropy_for_feature(feature_dict):
    """
    Args:
        feature_dict: {0: {0: 0, 1: 0},  # {x = 0: dict(y), ..}
                       1: {0: 0, 1: 0}}
    Returns:
        a list of entropy values, (+, -) for the feature, xTrains.
    """
    entropies = []
    for feature in [0, 1]:
        try:
            ys = feature_dict[feature]
        except TypeError as te:
            raise te
        y0s, y1s = ys[0], ys[1]
        y_total = y0s + y1s
        if y_total == 0:
            p0 = float((y0s + 1) / (y_total + 2))
            p1 = float((y1s + 1) / (y_total + 2))
        else:
            p0 = float(y0s / y_total)
            p1 = float(y1s / y_total)

        if p0 == 1.0 or p1 == 1.0:
            entropies.append(0.0)
            continue
        try:
            entropies.append(-p0 * math.log2(p0) - p1 * math.log2(p1))
        except ValueError as ve:
            raise ve

    return entropies


def get_feature_dict(xTrains, yTrains):
    """
    :param xTrains: a list of feature i values, [x for x in zip(*xTrains)][i]
    :param yTrains: a list of target attributes
    :return
        a list of entropy values, (+, -) for the feature, xTrains.
    """
    if not len(xTrains) == len(yTrains):
        raise ValueError("Unmatched list lengths {} vs {}".format(len(xTrains), len(yTrains)))

    feature_dict = {0: {0: 0, 1: 0},  # {x = 0: dict(y), ..}
                    1: {0: 0, 1: 0}}

    if all(yTrains):  # all ys are 1
        return feature_dict

    if not any(yTrains):  # all ys are 0
        return feature_dict

    if yTrains.count(1) == yTrains.count(0):  # same counts for target attributes
        return feature_dict

    feature_threshold = None
    if not set(xTrains) == set([0, 1]):
        feature_threshold = (max(xTrains) - min(xTrains)) / 2

    for x, y in zip(xTrains, yTrains):
        if feature_threshold:
            feature_dict[int(feature_threshold <= x)][y] += 1
        else:
            feature_dict[x][y] += 1

    return feature_dict


def get_information_gain(xTrains, yTrains):
    """
    Loss(S, 𝑥_𝑖)
     lossSum = 0
     for 𝑆_𝑣𝑎𝑙 in SplitByFeatureValues(S, 𝑥_𝑖)
          lossSum += Entropy(𝑆_𝑣𝑎𝑙 ) * len(𝑆_𝑣𝑎𝑙)
     return lossSum / len(S)

    InformationGain(S, 𝑥_𝑖)
     return Entropy(S) – Loss(S, 𝑥_𝑖)


    :param xTrains: a list of feature i values, [x for x in zip(*xTrains)][i]
    :param yTrains: a list of target attributes
    :return: information gain for the feature
    """
    feature_dict = get_feature_dict(xTrains, yTrains)
    entropies = get_entropy_for_feature(feature_dict)
    entropy_S = get_entropy_S(yTrains)
    loss_sum = 0
    for k, ys in feature_dict.items():
        loss_sum += entropies[k] * sum(ys.values())
    loss = loss_sum / len(yTrains)
    return entropy_S - loss


def get_entropy_S(yTrains):
    """
    Calculate entropy of S, yTrains. Mitchell page 57, eq(3.3)

    In tree, yTrains will be the splitted yTrains."""
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
    """
    Calculates information gains for all features.
    :param xTrains: examples from training data,
                    [[outlook_0, temperature_0, humidity_0, wind_0],[],...]
    :param yTrains: a list of target attribute, [0, 1, 0, 0,...]
    :return: a list of information gains for features,
            i.e., outlook, temperature, humidity, wind.
    """
    gains = []
    for xTrain in zip(*xTrains):
        gains.append(get_information_gain(xTrain, yTrains))
    return gains


def get_split(xTrains, yTrains):
    """
    Split dataset and create a node based on the feature [i] who has highest information gain.

    :param xTrains: examples from training data,
                    [[outlook_0, temperature_0, humidity_0, wind_0],[],...]
    :param yTrains: a list of target attribute, [0, 1, 0, 0,...]

    :returns
        an instance of Node dict {'index': the index of feature selected,
                                  'gain': info. gain,
                                  'groups': [((examples), (ys)), ((examples), (ys))]}
    """
    i_gails = get_information_gains(xTrains, yTrains)
    feature_index = i_gails.index(max(i_gails))
    threshold = get_feature_split_threshold(feature_index, xTrains)
    groups = split_by_feature(feature_index, xTrains, yTrains)

    return {'index': feature_index, 'gain': max(i_gails), 'groups': groups,
            'num_label_1': groups[0][1].count(1),
            'num_label_0': groups[0][1].count(0),
            'threshold': threshold}


def get_feature_split_threshold(feature_index, xTrains):
    # gather all values of feature xTrain[i]
    values_by_features = [x for x in zip(*xTrains)]
    unique_values = list(set(values_by_features[feature_index]))
    if set(unique_values) == set([0, 1]):
        return 0.5
    return (max(unique_values) - min(unique_values)) / 2


def split_by_feature(feature_index, xTrains, yTrains, threshold=None):
    """
    Calculates the threshold based on the feature values at i and split data
    into two groups.
    :param xTrains: examples from training data,
                    [[outlook_0, temperature_0, humidity_0, wind_0],[],...]
    :param yTrains: a list of target attribute, [0, 1, 0, 0,...]
    :returns:
        ((l_xTrains, l_yTrains), (r_xTrains, r_yTrains))
    """
    if not len(xTrains) == len(yTrains):
        raise ValueError("Unmatched xTrains({}) and yTrains({})".format(
            len(xTrains), len(yTrains)))

    if feature_index > len(xTrains):
        raise IndexError(
            "Feature index({}) is outside of xTrains".format(feature_index))

    if not threshold:
        threshold = get_feature_split_threshold(feature_index, xTrains)

    l_xTrains, l_yTrains = [], []
    r_xTrains, r_yTrains = [], []
    for xTrain, yTrain in zip(xTrains, yTrains):
        if xTrain[feature_index] >= threshold:
            l_xTrains.append(xTrain)
            l_yTrains.append(yTrain)
        else:
            r_xTrains.append(xTrain)
            r_yTrains.append(yTrain)

    print("{} samples are splitted into L({}), R({})".format(
        len(xTrains), len(l_xTrains), len(r_xTrains)))
    return (l_xTrains, l_yTrains), (r_xTrains, r_yTrains)


def split(node, min_to_stop=100):
    ((l_xTrains, l_yTrains), (r_xTrains, r_yTrains)) = node['groups']
    del (node['groups'])

    if not l_xTrains or not r_xTrains:
        ys = l_yTrains + r_yTrains
        node['left'] = node['right'] = Counter(ys).most_common(1)[0][0]
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


def build_tree(xTrains, yTrains, min_to_stop=100):
    root = get_split(xTrains, yTrains)
    split(root, min_to_stop)
    return root


# Print a decision tree
def print_tree(node, depth=0, indent='...,'):

    if isinstance(node, dict):
        print('{}Feature {}: '.format(str(depth * indent), node['index']))
        num_label_1 = node['num_label_1']
        num_label_0 = node['num_label_0']
        print('{}...,>= {}:'.format(str(depth * indent), node['threshold']))
        print('{}...,Leaf: {} vs {}'.format(str((depth + 1) * indent), num_label_1, num_label_0))
        print('{}...,< {}:'.format(str(depth * indent), node['threshold']))
        if 'left' in node:
            print_tree(node['left'], depth+1)
        if 'right' in node:
            print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % (depth * indent, node))


def write_tree(node, file_obj, depth=0, indent='...,'):

    if isinstance(node, dict):
        file_obj.write('\n{}Feature {}: '.format(str(depth * indent), node['index']))
        num_label_1 = node['num_label_1']
        num_label_0 = node['num_label_0']
        file_obj.write('\n{}...,>= {}:'.format(str(depth * indent), node['threshold']))
        file_obj.write('\n{}...,Leaf: {} vs {}'.format(str((depth + 1) * indent), num_label_1, num_label_0))
        file_obj.write('\n{}...,< {}:'.format(str(depth * indent), node['threshold']))
        if 'left' in node:
            write_tree(node['left'], file_obj, depth + 1)
        if 'right' in node:
            write_tree(node['right'],  file_obj, depth + 1)
    else:
        file_obj.write('\n%s[%s]' % (depth * indent, node))


def predict_w_threshold(node, example, threshold=0.5):
    """
    Predict the value for an example given a tree(/node):
         {'index': feature_index, 'gain': max(i_gails), 'groups': groups,
                'num_label_1': groups[0][1].count(1),
                'num_label_0': groups[0][1].count(0),
          'threshold': 0.5 or midpoint of features}
    :param node:
    :param row:
    :return:
    """
    if example[node['index']] >= node['threshold']:
        if isinstance(node['left'], dict):
            return predict_w_threshold(node['left'], example, threshold)
        else:
            if node['left'] == 1:
                return int(node['num_label_1'] / (node['num_label_1'] + node['num_label_0']) < threshold)
            return int(node['num_label_0'] / (node['num_label_1'] + node['num_label_0']) < threshold)
    else:
        if isinstance(node['right'], dict):
            return predict_w_threshold(node['right'], example, threshold)
        else:
            if node['right'] == 1:
                return int(node['num_label_1'] / (node['num_label_1'] + node['num_label_0']) < threshold)
            return int(node['num_label_0'] / (node['num_label_1'] + node['num_label_0']) < threshold)


def predict(node, example):
    """
    Predict the value for an example given a tree(/node):
         {'index': feature_index, 'gain': max(i_gails), 'groups': groups,
                'num_label_1': groups[0][1].count(1),
                'num_label_0': groups[0][1].count(0)}
    :param node:
    :param row:
    :return:
    """
    if example[node['index']] >= 0.5:
        if isinstance(node['left'], dict):
            return predict(node['left'], example)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], example)
        else:
            return node['right']