import collections as col
import hashlib
import numpy as np
import os
import pickle
from model import cache_dir
from sklearn.neighbors.kd_tree import KDTree

runtime_data = {}


class KNearestNeighborModel(object):

    def __init__(self, xTrains, yTrains):
        self.xTrains = xTrains
        self.yTrains = yTrains
        self.tree = KDTree(self.xTrains, leaf_size=2)


    def get_ordered_train_neighbors(self, xTests):

        hash_str = '{}_{}'.format(self.xTrains, xTests).encode()
        knn_hash = hashlib.sha256(hash_str).hexdigest()
        knn_pkl = os.path.join(cache_dir, '{}.pkl'.format(knn_hash))
        if os.path.exists(knn_pkl):
            neighbors_ordered = runtime_data.get(knn_hash, None)
            if not neighbors_ordered:
                print("Found pickled xTrains sorted.")
                with open(knn_pkl, 'rb') as f:
                    neighbors_ordered = pickle.load(f)
                    runtime_data[knn_hash] = neighbors_ordered
            return neighbors_ordered

        neighbors_ordered = []
        for xTest in xTests:

            distances = []
            for xt, yt in zip(self.xTrains, self.yTrains):
                a1 = np.array(xTest)
                a2 = np.array(xt)
                d = np.linalg.norm(a1 - a2)
                if d == 0:
                    continue
                distances.append((d, xt, yt))

            neighbors_ordered.append(sorted(distances))

        KNearestNeighborModel.runtime_data[knn_hash] = neighbors_ordered
        with open(knn_pkl, 'wb') as f:
            pickle.dump(neighbors_ordered, f)
            print("Saved neighbors ordered in pickle {}".format(knn_pkl))

        return neighbors_ordered

    def predict_old(self, xTests, k, threshold=None):

        if k > len(self.xTrains):
            k = len(self.xTrains)

        predictions = []
        neighbors_ordered = self.get_ordered_train_neighbors(xTests)
        for neighbors in neighbors_ordered:
            k_neighbors = neighbors[:k]

            k_predictions = [sd[-1] for sd in k_neighbors]

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
