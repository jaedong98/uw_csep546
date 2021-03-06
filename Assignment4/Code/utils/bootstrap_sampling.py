import collections
import random
import numpy as np


def get_bagging_indices(sample_size, seed=0):
    """

    :param sample_size: total number of sample size, len(xTrains)
    :param seed: seed number for random generation
    :return: a list of indices randomly generated in range(0, sample_size)
    """
    bagging_indices = []
    if seed is not None:
        random.seed(seed)
    for _ in range(sample_size):
        bagging_indices.append(random.randint(0, sample_size - 1))

    if not len(bagging_indices) == sample_size:
        raise AssertionError("Unmatched bagging size {} vs {}"
                             .format(len(bagging_indices), sample_size))
    return bagging_indices


def get_bagged_samples_pre(samples, seed=0):
    """
    Bootstraptes samples.

    :param samples: a list of samples, either xTrains or xTests
    :param seed:
    :return:
    """
    bagging_indices = get_bagging_indices(len(samples), seed)
    new_samples = []
    for i in bagging_indices:
        new_samples.append(samples[i])

    return new_samples


def get_bagged_samples(samples, seed=0):
    """
    Bootstraptes samples.

    :param samples: a list of samples, either xTrains or xTests
    :param seed:
    :return:
    """
    ss = len(samples)
    if seed is not None:
        np.random.seed(seed)
    bagging_indices = np.random.choice(ss, ss)
    #bagging_indices = [random.randint(0, ss - 1) for _ in range(ss)]
    mc = collections.Counter(bagging_indices).most_common(3)
    print("Most duplicated indices: {}".format(mc))
    return [samples[i] for i in bagging_indices]


def bootstraping_training_data(xTrains, yTrains, seed=0):
    """
    Bootstraptes samples.

    :param samples: a list of samples, either xTrains or xTests
    :param seed:
    :return:
    """
    ss = len(yTrains)
    if seed is not None:
        np.random.seed(seed)
    bagging_indices = np.random.choice(ss, ss)

    return [xTrains[i] for i in bagging_indices], [yTrains[i] for i in bagging_indices]

