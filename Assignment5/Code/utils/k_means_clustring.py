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

    def reset_centroids(self):
        [c.reset_samples() for c in self.centroids]

    def cluster(self):

        for i in range(self.iterations):
            print("Iteration = {}".format(i))
            self.reset_centroids()
            for sample in self.samples:

                cent_i = sample.find_centroid(self.centroids)
                self.centroids[cent_i] = self.centroids[cent_i].add_sample(sample)

            self.centroids = [c.update() for c in self.centroids]

    def closest_pairs(self):

        return [(c, c.closest_sample()) for c in self.centroids]


class Centroid(object):

    def __init__(self, features, samples=[], prev=None):
        self.x, self.y = features
        self.samples = samples
        if prev:
            self.logs = prev.logs + [(self.x, self.y)]
        else:
            self.logs = [(self.x, self.y)]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def add_sample(self, sample):
        return Centroid([self.x, self.y], self.samples + [sample], self)

    def reset_samples(self):
        self.samples = []

    def update(self):

        xs, ys = [x for x in zip(*[[s.x, s.y] for s in self.samples])]
        features = [sum(xs) / len(xs), sum(ys) / len(ys)]
        print("{} Samples - Moving from [{},{}] to [{},{}]"
              .format(len(self.samples), self.x, self.y, features[0], features[1]))
        return Centroid(features, list(self.samples), self)

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

    def path_xs_ys(self):
        return [x for x in zip(*self.logs)]

    def sample_xs_ys(self):
        if not self.samples:
            return [], []
        samples = [(s.x, s.y) for s in self.samples]
        return [x for x in zip(*samples)]

    def __repr__(self):
        return "[{},{}]".format(self.x, self.y)


class Sample(object):

    def __init__(self, features):
        self.x, self.y = features

    def find_centroid(self, centroids):

        min_d = float('inf')
        min_cent_index = None
        for i, c in enumerate(centroids):
            d = np.linalg.norm(np.array([self.x, self.y] - np.array([c.x, c.y])))
            if d < min_d:
                min_d = d
                min_cent_index = i

        return min_cent_index

    def __repr__(self):
        return "[{},{}]".format(self.x, self.y)


