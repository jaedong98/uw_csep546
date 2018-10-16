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
                                     initial_w0=0.0, report_path=report_path, fname='',
                                     predict_thresholds=[0.5],
                                     additional_features=[]):
    
    # features by MI
    features = fbm.extract_features_by_mi(xTrainRaw, yTrainRaw, N)

    # gradient decent
    iter_cnts = [0]
    resolution = int(max_iters / iter_step)
    features = [x[0] for x in features]

    if additional_features:
        features = list(set(features + additional_features))

    if not fname:
        fname = 'precision_vs_recall_{}_thresholds.png'.format(len(predict_thresholds))
    img_fname = os.path.join(report_path, fname)
    
    title = "Precision vs Recall"
    legends = ["Precision", "Recall"]
    precisions = []
    recalls = []
    for predict_threshold in predict_thresholds:
        evaluation = utils.logistic_regression_by_features(xTrainRaw, xTestRaw,
                                                        yTrain, yTest,
                                                        features, iter_step,
                                                        resolution, initial_w0,
                                                        step, max_iters,
                                                        predict_threshold=predict_threshold,
                                                        return_last_evals=True)
        precisions.append((predict_threshold, evaluation.precision))
        recalls.append((predict_threshold, evaluation.recall))

    utils.draw_comparison([precisions, recalls], "Thresholds", "Precision vs. Recalls", title, img_fname, legends)

if __name__ == '__main__':

    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)

    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    predict_thresholds = list(np.linspace(0, 1, 11))


    N = 10
    fname = 'precision_vs_recall_{}_thresholds.png'.format(len(predict_thresholds))
    run_logistic_regression_w_threshold(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                        max_iters=50000, iter_step=1000, step=0.01,
                                        initial_w0=0.0, fname=fname,
                                        predict_thresholds=predict_thresholds)
    
    fname = 'precision_vs_recall_{}_thresholds_w_additional.png'.format(len(predict_thresholds))
    run_logistic_regression_w_threshold(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                        max_iters=50000, iter_step=1000, step=0.01,
                                        initial_w0=0.0, fname=fname,
                                        predict_thresholds=[x for x in range(0, 1, 0.1)],
                                        additional_features=['call', 'to', 'your'])