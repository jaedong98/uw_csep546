import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import features_by_frequency as fbf
import features_by_mi as fbm


import Assignment2Support as utils
import EvaluationsStub
import LogisticRegressionModel as lgm

# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def run_comparision(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, N, max_iters, iter_step, step, initial_w0, features=[], report_path=report_path):
    losses = []
    accuracies = []
    legends = []
    # Top N frequency
    iter_cnt_vs_loss, iter_cnt_vs_accuracy, features_by_frequency = fbf.run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                                                                             max_iters=max_iters, iter_step=iter_step, step=step,
                                                                                             initial_w0=initial_w0)
    losses.append(iter_cnt_vs_loss)
    accuracies.append(iter_cnt_vs_accuracy)
    legends.append('Top {} Frequency'.format(N))

    # Top N MI
    iter_cnt_vs_loss, iter_cnt_vs_accuracy, features_by_mi = fbm.run_gradient_descent(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                                                                      max_iters=max_iters, iter_step=iter_step, step=step,
                                                                                      initial_w0=initial_w0)
    losses.append(iter_cnt_vs_loss)
    accuracies.append(iter_cnt_vs_accuracy)
    legends.append('Top {} MI'.format(N))

    # Merged Features:
    merged_features = list(set(features_by_frequency + features_by_mi))
    iter_cnts = [0]
    resolution = int(max_iters / iter_step)
    img_fname = os.path.join(
        report_path, 'iter_cnt_vs_accuracy_by_merged_features_{}.png'.format(max_iters))
    title = "Accuracy Over Iteration by Merged Features."
    iter_cnt_vs_loss, iter_cnt_vs_accuracy = utils.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                                                   yTrain, yTest,
                                                                                   merged_features, iter_step,
                                                                                   resolution, initial_w0,
                                                                                   step, max_iters, img_fname, title)
    losses.append(iter_cnt_vs_loss)
    accuracies.append(iter_cnt_vs_accuracy)
    legends.append('Merged Features*')

    # Custom features:
    if features:
        iter_cnts = [0]
        resolution = int(max_iters / iter_step)
        img_fname = os.path.join(
            report_path, 'iter_cnt_vs_accuracy_by_custom_features_{}.png'.format(max_iters))
        title = "Accuracy Over Iteration by Custom Features."
        iter_cnt_vs_loss, iter_cnt_vs_accuracy = utils.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                                                       yTrain, yTest,
                                                                                       features, iter_step,
                                                                                       resolution, initial_w0,
                                                                                       step, max_iters, img_fname, title)
        losses.append(iter_cnt_vs_loss)
        accuracies.append(iter_cnt_vs_accuracy)
        legends.append('Custom Features*')

    # Outputs
    img_fname = os.path.join(
        report_path, 'accuracy_comparison_top_{}.png'.format(N))

    utils.draw_accuracies(accuracies, 'Iterations', 'Accuracy',
                          'Accuracy Comparision', img_fname, legends)

    table = utils.table_for_gradient_accuracy_comparision(accuracies, legends)
    table_md = os.path.join(
        report_path, 'accuracy_comparision_by_feature_selections_{}.md'.format(N))
    print(table)
    with open(table_md, 'w') as f:
        f.write(table)
        f.write('\n')
        f.write('\nMerged Features selected:')
        f.write('\n{}'.format(merged_features))

        if features:
            f.write('\n')
            f.write('\nCustom Features selected:')
            f.write('\n{}'.format(features))


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    # Configuration
    max_iters = 50000
    iter_step = 1000
    step = 0.01
    initial_w0 = 0.0

    # Top 10
    N = 10
    features = ['I', 'Call', 'i', 'Free',
                'claim', 'to', 'you', 'a', 'the', 'and',
                'prize', 'www.', 'customer']
    #run_comparision(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
    #                N, max_iters, iter_step, step, initial_w0, features)

    # Top 100
    N = 100
    run_comparision(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                    N, max_iters, iter_step, step, initial_w0, features)
