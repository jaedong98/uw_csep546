import numpy as np
import random


class KMeanClustring(object):

    def __init__(self, xTrains, k, iterations):

        self.samples = [Sample(x) for x in xTrains]
        self.k = k
        self.iterations = iterations

        random.seed(100)
        sample_indices = random.sample(range(len(xTrains)), k)
        self.centroids = [Centroid(xTrains[i]) for i in sample_indices]

    def cluster(self):

        for i in range(self.k):

            for sample in self.samples:

                sample.find_centroid(self.centroids)

            self.centroids = [c.update() for c in self.centroids]

    def closest_pairs(self):

        return [(c, c.closest_sample()) for c in self.centroids]


class Centroid(object):

    def __init__(self, features, samples=[], prev=None):
        self.x, self.y = features
        self.samples = samples
        if prev:
            self.logs = [prev, self]
        self.logs = [self]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def update(self):

        xs, ys = [x for x in zip(*[[s.x, s.y] for s in self.samples])]
        features = [sum(xs) / len(xs), sum(ys) / len(ys)]
        print("Moving from [{},{}] to [{},{}]"
              .format(self.x, self.y, features[0], features[1]))
        return Centroid(features, self.samples, self)

    def closest_sample(self):

        min_d = float('inf')
        closest_sample = None
        for s in self.samples:
            d = np.linalg.norm(np.array([self.x, self.y] - np.array([s.x, s.y])))
            if d < min_d:
                min_d = d
                closest_sample = s

        return closest_sample

    def path(self):
        return [(c.x, c.y) for c in self.logs]

    def __repr__(self):
        return "[{},{}]".format(self.x, self.y)


class Sample(object):

    def __init__(self, features):
        self.x, self.y = features

    def find_centroid(self, centroids):

        min_d = float('inf')
        min_cent = None
        for c in centroids:
            d = np.linalg.norm(np.array([self.x, self.y]- np.array([c.x, c.y])))
            if d < min_d:
                min_d = d
                min_cent = c

        min_cent.samples.append(self)

    def __repr__(self):
        return "[{},{}]".format(self.x, self.y)


