import os
from Assignment5.Code import kDataPath, report_path
from utils.Assignment5Support import LoadRawData, TrainTestSplit, Featurize
from utils.k_means_clustring import KMeanClustring

import matplotlib.pyplot as plt


def plot_data_and_centroid_paths(xTrains, k, iterations):

    colors = ['b', 'r', 'g', 'y', 'm', 'c'] * k

    kmc = KMeanClustring(xTrains, k, iterations)
    kmc.cluster()

    for i, (color, centroid) in enumerate(zip(colors, kmc.centroids)):
        xs, ys = centroid.path_xs_ys()
        plt.show()  # erase
        plt.plot(xs, ys, '-+')

        sample_xs, sample_ys = centroid.sample_xs_ys()
        if not sample_xs and not sample_ys:
            continue
        plt.scatter(sample_xs, sample_ys, c=color, alpha=0.2)
        plt.title("Clusterting K={}, {}th Centroid".format(k, i))
        plt.xlabel("average Y gradient strength")
        plt.ylabel("average Y gradient strength in middle 3rd")
        fname = 'prob2_plot_data_and_centroid_paths_centroid{}(K={}).png'.format(i, k)
        fpath = os.path.join(report_path, fname)
        plt.savefig(fpath)

    # full shot
    for i, (color, centroid) in enumerate(zip(colors, kmc.centroids)):
        xs, ys = centroid.path_xs_ys()
        plt.plot(xs, ys, '-+')

        sample_xs, sample_ys = centroid.sample_xs_ys()
        if not sample_xs and not sample_ys:
            continue
        plt.scatter(sample_xs, sample_ys, c=color, alpha=0.2)
        plt.title("Clusterting K={}, {}th Centroid".format(k, i))
        plt.xlabel("average Y gradient strength")
        plt.ylabel("average Y gradient strength in middle 3rd")

    fname = 'prob2_plot_data_and_centroid_paths_k{}_iteration{}.png'.format(k, iterations)
    fpath = os.path.join(report_path, fname)
    plt.savefig(fpath)
    plt.show()
    fname = os.path.join(report_path, 'prob2_closest_samples.md')
    with open(fname, 'w') as f:
        for pair in kmc.closest_pairs():
            f.write('\n* Centroid: {} - Sample {}'.format(pair[0], pair[1]))

    # closest_points
    for i, (color, centroid) in enumerate(zip(colors, kmc.centroids)):
        plt.show()
        cs = centroid.closest_sample()
        plt.scatter([centroid.x, cs.x], [centroid.y, cs.y], s=5, c='black', alpha=0.8)
        plt.plot([centroid.x, cs.x], [centroid.y, cs.y], 'black')
        sample_xs, sample_ys = centroid.sample_xs_ys()
        if not sample_xs and not sample_ys:
            continue
        plt.scatter(sample_xs, sample_ys, c=color, alpha=0.1)
        plt.title("Clusterting K={}, {}th Centroid".format(k, i))
        plt.xlabel("average Y gradient strength")
        plt.ylabel("average Y gradient strength in middle 3rd")
        fname = 'prob2_closest_sample_and_centroid{}(K={}).png'.format(i, k)
        fpath = os.path.join(report_path, fname)
        plt.savefig(fpath)


if __name__ == "__main__":
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)
    (xTrains, _) = Featurize(xTrainRaw, xTestRaw, includeGradients=True)
    k = 4
    iterations = 10
    plot_data_and_centroid_paths(xTrains, k, iterations)