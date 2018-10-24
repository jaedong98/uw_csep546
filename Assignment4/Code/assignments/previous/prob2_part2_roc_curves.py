import inspect
import os

import numpy as np

import utils.Assignment3Support as sup
import utils.EvaluationsStub as es
import model.DecisionTreeModel as dtm

"""
What is an ROC curve?

Ans. plot ( sensitivity vs (1 - specificity ) ) !!

Let's assume, you have built a Logistic Regression model.

1. while predicting, you need to give a threshold and based on that you'll get the predicted output and from that you can calculate sensitivity & specificity.
2. Now, go back to the predicting step and give some 10 threshold values from 0 to 1.
3. So, you have 10 sensitivity & specificity values!!
4. Arrange them in the increasing order of (1-specificity).
5. Draw the plot using these values.
"""
# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                   min_to_step,
                   threshold,
                   featurize=sup.Featurize):

    yTrain = yTrainRaw
    yTest = yTestRaw

    (xTrain, xTest) = featurize(xTrainRaw, xTestRaw)

    model = dtm.DecisionTreeModel()
    model.fit(xTrain, yTrain, min_to_step)

    yTestPredicted = model.predict(xTest, threshold)
    return es.Evaluation(yTest, yTestPredicted)


def compare_roc_curves_by_min_to_stop(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                      thresholds,
                                      report_path=report_path):

    graphs = []
    legends = []
    original_fpr_fnr = []

    for min_to_step in [420]:
        cont_length_fpr_fnr = []
        for threshold in thresholds:
            ev = get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                min_to_step=min_to_step,
                                threshold=threshold,
                                featurize=sup.FeaturizeWNumericFeature)
            if threshold == 0:
                assert (ev.fpr == 0.0), ev
            print(ev)
            cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
        graphs.append(cont_length_fpr_fnr)
        legends.append('Cont. Length {} minToSteps'.format(min_to_step))

    for threshold in thresholds:
        ev = get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                            min_to_step=100,
                            threshold=threshold,
                            featurize=sup.Featurize)
        if threshold == 0:
            assert(ev.fpr == 0.0), ev
        print(ev)
        original_fpr_fnr.append((ev.fpr, ev.fnr))
    graphs.append(original_fpr_fnr)
    legends.append('0/1 Length Feature')

    # plotting
    start = thresholds[0]
    end = thresholds[-1]
    step = thresholds[1] - thresholds[0]
    fname = '{}_{}_{}_{}.png'.format(inspect.stack()[0][3], start, end, step)
    img_fname = os.path.join(report_path, fname)
    sup.draw_accuracies(graphs,
                          'False Positive Rate', 'False Negative Rate', '',
                          img_fname,
                          legends=legends,
                          invert_yaxis=True)


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = sup.LoadRawData(kDataPath)

    # Train-Test split
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = sup.TrainTestSplit(xRaw, yRaw)

    start = 0
    end = 1
    N = 1000
    thresholds = [x / N for x in range(N + 1)]
    compare_roc_curves_by_min_to_stop(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                      thresholds,
                                      report_path=report_path)