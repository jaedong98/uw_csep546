import inspect
import os
from Assignment5.Code import kDataPath, report_path
from model.KnnModel import KNearestNeighborModel
from utils.Assignment4Support import draw_accuracies
from utils.Assignment5Support import Featurize, LoadRawData, TrainTestSplit
from utils.EvaluationsStub import Evaluation


def roc_curves_cmp(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs, thresholds):

    cnames = []
    for config in configs:
        cnames.append(config['name'])
        del config['name']

    for k in [1, 3, 5, 10, 20, 50, 100]:

        print("Predicting with K={}".format(k))
        fname = 'prob3{}_K{}_{}_thresholds.png' \
            .format(inspect.stack()[0][3], k, len(thresholds))
        graphs = []
        legends = cnames
        for cname, config in zip(cnames, configs):
            print("Predicting for {}".format(cname))
            (xTrains, xTests) = Featurize(xTrainRaw, xTestRaw, config)
            yTrains = yTrainRaw
            yTests = yTestRaw
            knn = KNearestNeighborModel(xTrains, yTrains)

            cont_length_fpr_fnr = []
            for threshold in thresholds:
                yTestPredicted = knn.predict(xTests, k, threshold)
                ev = Evaluation(yTests, yTestPredicted)
                cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
            graphs.append(cont_length_fpr_fnr)

        # plotting
        img_fname = os.path.join(report_path, fname)
        draw_accuracies(graphs,
                        'False Positive Rate', 'False Negative Rate',
                        'ROC Curves for K={}'.format(k),
                        img_fname,
                        legends=legends,
                        invert_yaxis=True,
                        data_pt='-')


if __name__ == '__main__':
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)

    config0 = {'name': 'grid_y_gradients',
               'grid_y_gradients': True,
               'grid_x_gradients': False,
               'hist_y_gradients': False,
               'hist_x_gradients': False}
    config1 = {'name': 'grid_x_gradients',
               'grid_y_gradients': False,
               'grid_x_gradients': True,
               'hist_y_gradients': False,
               'hist_x_gradients': False}
    config2 = {'name': 'hist_y_gradients',
               'grid_y_gradients': False,
               'grid_x_gradients': False,
               'hist_y_gradients': True,
               'hist_x_gradients': False}
    config3 = {'name': 'hist_x_gradients',
               'grid_y_gradients': False,
               'grid_x_gradients': False,
               'hist_y_gradients': False,
               'hist_x_gradients': True}

    N = 10
    thresholds = [x / N for x in range(N + 1)]
    configs = [config0, config1, config2, config3]
    roc_curves_cmp(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, configs, thresholds)