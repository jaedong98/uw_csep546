import collections
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import time

import Assignment3Support as utils
import EvaluationsStub
import LogisticRegressionModel as lgm

import features_by_mi as fbm

# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def run_logistic_regression_w_threshold(xTrainRaw, xTestRaw, yTrain, yTest, N=10,
                                        max_iters=50000, iter_step=1000, step=0.01,
                                        initial_w0=0.0, report_path=report_path,
                                        predict_thresholds=[0.5],
                                        additional_features=[]):

    # features by MI
    features, _ = fbm.extract_features_by_mi(xTrainRaw, yTrainRaw, N)

    # gradient decent
    iter_cnts = [0]
    resolution = int(max_iters / iter_step)
    features = [x[0] for x in features]

    if additional_features:
        features = list(set(features + additional_features))

    xTest = utils.FeaturizeTrainingByWords(xTestRaw, features)
    xTest = [[1] + x for x in xTest]
    model = utils.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                  yTrain, yTest,
                                                  features, iter_step,
                                                  resolution, initial_w0,
                                                  step, max_iters,
                                                  return_model=True)
    precisions_vs_recalls = []
    thresholds_fp = []
    thresholds_fn = []
    for predict_threshold in predict_thresholds:

        yTestPredicted = model.predict(xTest, predict_threshold)
        evaluation = EvaluationsStub.Evaluation(yTest, yTestPredicted)
        precisions_vs_recalls.append((evaluation.recall, evaluation.precision))
        thresholds_fp.append((predict_threshold, evaluation.fpr))
        thresholds_fn.append((predict_threshold, evaluation.fnr))
    return precisions_vs_recalls, thresholds_fp, thresholds_fn


if __name__ == '__main__':

    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)

    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw

    N = 10
    max_iters = 50000
    iter_step = 1000
    step = 0.01
    initial_w0 = 0.0
    predict_thresholds = list(np.linspace(0.01, 0.99, 101))
    additional_features = ['call', 'to', 'your']
    ##
    prs1, fp1, fn1 = run_logistic_regression_w_threshold(
        xTrainRaw, xTestRaw, yTrain, yTest,
        N=N,
        max_iters=max_iters,
        iter_step=iter_step,
        step=step,
        initial_w0=initial_w0,
        predict_thresholds=predict_thresholds)
    csv_fname = os.path.join(
        report_path, 'precision_vs_recall_{}_thresholds.csv'.format(max_iters))
    utils.write_csv(prs1, csv_fname, headers=["Recall", "Precision"])
    csv_fname = os.path.join(
        report_path, 'thresholds_fp_{}.csv'.format(max_iters))
    utils.write_csv(fp1, csv_fname, headers=[
                    "Thresholds", "False Positive Rates"])
    csv_fname = os.path.join(
        report_path, 'thresholds_fn_{}.csv'.format(max_iters))
    utils.write_csv(fn1, csv_fname, headers=[
                    "Thresholds", "False Negative Rates"])
    ##
    prs2, fp2, fn2 = run_logistic_regression_w_threshold(
        xTrainRaw, xTestRaw, yTrain, yTest,
        N=N,
        max_iters=max_iters,
        iter_step=iter_step,
        step=step,
        initial_w0=initial_w0,
        predict_thresholds=predict_thresholds,
        additional_features=additional_features)
    csv_fname = os.path.join(
        report_path, 'precision_vs_recall_{}_thresholds_w_additional.csv'.format(len(predict_thresholds)))
    utils.write_csv(prs2, csv_fname, headers=["Recall", "Precision"])
    csv_fname = os.path.join(
        report_path, 'thresholds_fp_{}_w_additional.csv'.format(max_iters))
    utils.write_csv(fp2, csv_fname, headers=[
                    "Thresholds", "False Positive Rates"])
    csv_fname = os.path.join(
        report_path, 'thresholds_fn_{}_w_additional.csv'.format(max_iters))
    utils.write_csv(fn2, csv_fname, headers=[
                    "Thresholds", "False Negative Rates"])

    # problem 1 part 1.
    title = "Precision vs. Recalls"
    legends = ['Top 10 MI', 'Top 10 MI with +3']
    fname = 'precision_vs_recall_{}.png'.format(max_iters)
    img_fname = os.path.join(report_path, fname)
    utils.draw_comparison([prs1, prs2], "Recalls",
                          "Precisions", title, img_fname, legends)

    # problem 1 part2.
    title = "Thresholds vs. FPRs"
    legends = ['Top 10 MI', 'Top 10 MI with +3']
    fname = 'thresholds_fpr_comp_{}.png'.format(max_iters)
    img_fname = os.path.join(report_path, fname)
    utils.draw_comparison([fp1, fp2], "Thresholds",
                          "False Positive Rates", title, img_fname, legends)

    title = "Thresholds vs. FNRs"
    legends = ['Top 10 MI', 'Top 10 MI with +3']
    fname = 'thresholds_fnr_comp_{}.png'.format(max_iters)
    img_fname = os.path.join(report_path, fname)
    utils.draw_comparison([fn1, fn2], "Thresholds",
                          "False Negative Rates", title, img_fname, legends)
