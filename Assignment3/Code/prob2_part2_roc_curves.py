import inspect
import os

import numpy as np

import Assignment3Support as utils
import EvaluationsStub as es
import DecisionTreeModel as dtm


# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")


def get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                   min_to_step,
                   threshold,
                   featurize=utils.Featurize):

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
    for threshold in thresholds:
        ev = get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                            min_to_step=100,
                            threshold=threshold,
                            featurize=utils.Featurize)
        original_fpr_fnr.append((ev.fpr, ev.fnr))
    graphs.append(original_fpr_fnr)
    legends.append('0.1 Length Feature')

    for min_to_step in [450, 500, 550]:
        cont_length_fpr_fnr = []
        for threshold in thresholds:
            ev = get_evaluation(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                min_to_step=min_to_step,
                                threshold=threshold,
                                featurize=utils.FeaturizeWNumericFeature)
            print(ev)
            cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
        graphs.append(cont_length_fpr_fnr)
        legends.append('Cont. Length {} minToSteps'.format(min_to_step))

    start = thresholds[0]
    end = thresholds[-1]
    step = thresholds[1] - thresholds[0]
    fname = '{}_{}_{}_{}.png'.format(inspect.stack()[0][3], start, end, step)
    img_fname = os.path.join(report_path, fname)
    utils.draw_accuracies(graphs,
                          'False Positive Rate', 'False Negative Rate', 'ROC comparision',
                          img_fname,
                          legends=legends)

    cm_md = os.path.join(report_path, fname.replace('.png', '.md'))
    #with open(cm_md, 'w') as f:
    #    f.write(table)


if __name__ == '__main__':
    # Loading data
    (xRaw, yRaw) = utils.LoadRawData(kDataPath)

    # Train-Test split
    (xTrainRaw, yTrainRaw, xTestRaw,
     yTestRaw) = utils.TrainTestSplit(xRaw, yRaw)

    start = 0.01
    end = 1
    N = 10
    thresholds = [t for t in np.linspace(start, end, N)]
    compare_roc_curves_by_min_to_stop(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw,
                                      thresholds,
                                      report_path=report_path)