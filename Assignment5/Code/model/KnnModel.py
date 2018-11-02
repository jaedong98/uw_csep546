import collections as col
from sklearn.neighbors.kd_tree import KDTree

runtime_data = {}


class KNearestNeighborModel(object):

    def __init__(self, xTrains, yTrains):
        self.xTrains = xTrains
        self.yTrains = yTrains
        self.tree = KDTree(self.xTrains, leaf_size=2)

    def predict(self, xTests, k, threshold=None):

        if k > len(self.xTrains):
            k = len(self.xTrains)

        predictions = []
        for xTest in xTests:
            _, ind = self.tree.query([xTest], k=k)
            k_predictions = [self.yTrains[i] for i in ind[0]]

            if threshold is not None:
                cnt_1 = k_predictions.count(1)
                proposition = cnt_1 / k
                if proposition >= threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                mc = col.Counter(k_predictions).most_common(1)[0][0]
                predictions.append(mc)

        return predictions
