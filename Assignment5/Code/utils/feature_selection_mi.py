import collections
import math


def calculate_mi(y0_counter, y1_counter, top=10):

    features = y0_counter.keys()
    y0_N = sum(y0_counter.values())
    y1_N = sum(y1_counter.values())
    N = y0_N + y1_N
    mi = collections.Counter()
    for f in features:
        if not f:
            continue
        # For 'Call'

        #          |   'call'  | no 'Call' | ...
        # |--------|-----------|-----------|...
        # | y(=1)  |    n11    |    n10    |...
        # | y(=0)  |    n01    |    n00    |...
        n11 = y1_counter[f]         # number of 'Call' when y = 1
        n10 = y1_N - y1_counter[f]  # number of no 'Call' when y = 1
        n01 = y0_counter[f]         # number of 'Call' when y = 0
        n00 = y0_N - y0_counter[f]  # number of no 'Call' when y = 0

        n = n00 + n01 + n10 + n11
        n1_ = n10 + n11
        n_1 = n01 + n11
        n0_ = n00 + n01
        n_0 = n00 + n10

        mi[f] = (n11/n) * math.log2((n*n11 + 1) / (n1_ * n_1))\
            + (n01/n) * math.log2((n*n01 + 1) / (n0_ * n_1))\
            + (n10/n) * math.log2((n*n10 + 1) / (n1_ * n_0))\
            + (n00/n) * math.log2((n*n00 + 1) / (n0_ * n_0))

    return mi.most_common(top)


def extract_features_by_mi(xTrainRaw, yTrainRaw, N):
    """
    Returns: a list of tuples, ('feature', mi value)
    """
    # MI calculation
    y0_counter = collections.Counter()
    y1_counter = collections.Counter()
    for y, x in zip(yTrainRaw, xTrainRaw):
        for w in x.split(' '):
            if y == 0:
                y0_counter[w] += 1
                y1_counter[w] += 0
            else:
                y0_counter[w] += 0
                y1_counter[w] += 1

    if not len(y0_counter.keys()) == len(y1_counter.keys()):
        raise ValueError('Missing keys() {} vs {}'.format(
            len(y0_counter.keys()), len(y1_counter.keys())))

    return calculate_mi(y0_counter, y1_counter, top=N)
