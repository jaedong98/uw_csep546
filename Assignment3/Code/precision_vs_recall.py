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
    for predict_threshold in predict_thresholds:
        
        yTestPredicted = model.predict(xTest, predict_threshold)
        evaluation = EvaluationsStub.Evaluation(yTest, yTestPredicted)
        precisions_vs_recalls.append((evaluation.recall, evaluation.precision))

    return precisions_vs_recalls



if __name__ == '__main__':

    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)

    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)
    yTrain = yTrainRaw
    yTest = yTestRaw
    predict_thresholds = list(np.linspace(0.01, 0.99, 101))
    max_iters = 500
    iter_step = 100
    step = 0.01
    initial_w0 = 0.0
    N = 10
    fname = 'precision_vs_recall_{}_thresholds.png'.format(len(predict_thresholds))
    prs1 = run_logistic_regression_w_threshold(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                        max_iters=max_iters, iter_step=iter_step, step=step,
                                        initial_w0=initial_w0, fname=fname,
                                        predict_thresholds=predict_thresholds)
    csv_fname = os.path.join(report_path, fname.replace('.png', '.csv'))
    utils.write_csv(prs1, csv_fname, headers=["Recall", "Precision"])

    fname = 'precision_vs_recall_{}_thresholds_w_additional.png'.format(len(predict_thresholds))
    prs2 = run_logistic_regression_w_threshold(xTrainRaw, xTestRaw, yTrain, yTest, N=N,
                                        max_iters=max_iters, iter_step=iter_step, step=step,
                                        initial_w0=initial_w0, fname=fname,
                                        predict_thresholds=predict_thresholds,
                                        additional_features=['call', 'to', 'your'])
    csv_fname = os.path.join(report_path, fname.replace('.png', '.csv'))
    utils.write_csv(prs2, csv_fname, headers=["Recall", "Precision"])

    title = "Precision vs. Recalls"
    legends = ['Top 10 MI', 'Top 10 MI with +3']
    fname = 'precision_vs_recall_{}.png'.format(max_iters)
    img_fname = os.path.join(report_path, fname)
    utils.draw_comparison([prs1, prs2], "Recalls", "Precisions", title, img_fname, legends)
