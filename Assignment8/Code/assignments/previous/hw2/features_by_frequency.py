import collections
import os
import utils.Assignment4Support as sup


# Loading data
from Assignment4.Code import kDataPath, report_path

(xRaw, yRaw) = sup.LoadRawData(kDataPath)


def extract_features_by_frequency(xTrainRaw, N):

    count = collections.Counter()
    for x in xTrainRaw:
        for w in x.split(' '):
            count[w] += 1

    return count.most_common(N)


def run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                         max_iters=50000, iter_step=1000, step=0.01,
                         initial_w0=0.0, report_path=report_path, fname=''):
    """
    Returns: iter_cnt_vs_loss, iter_cnt_vs_accuracy
    """
    features = extract_features_by_frequency(xTrainRaw, N)

    table = sup.selected_features_table(features, ["Features", "Frequency"])

    table_md = os.path.join(
        report_path, 'features_selected_by_top_{}_frequenct_words.md'.format(N))

    with open(table_md, 'w') as f:
        f.write(table)

    # gradient decent
    resolution = int(max_iters / iter_step)
    features = [x[0] for x in features]
    if not fname:
        fname = 'iter_cnt_vs_accuracy_by_frequency_{}_N{}.png'.format(max_iters, N)
    img_fname = os.path.join(report_path, fname)
    title = "Accuracy Over Iteration by Top {} Frequency Features.".format(N)
    iter_cnt_vs_loss, iter_cnt_vs_accuracy = sup.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                 yTrain, yTest,
                                                 features, iter_step,
                                                 resolution, initial_w0,
                                                 step, max_iters, img_fname, title)
    return iter_cnt_vs_loss, iter_cnt_vs_accuracy, features


if __name__ == '__main__':
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = sup.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                         max_iters=500, iter_step=100, step=0.01,
                         initial_w0=0.0)
