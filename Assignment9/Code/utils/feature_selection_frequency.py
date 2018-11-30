import collections


def extract_features_by_frequency(xTrainRaw, N):

    count = collections.Counter()
    for x in xTrainRaw:
        for w in x.split(' '):
            count[w] += 1

    return count.most_common(N)