import inspect
import os
from Assignment5.Code import kDataPath, report_path
from model.RandomForestsModel import RandomForestModel
from utils.Assignment4Support import draw_accuracies
from utils.Assignment5Support import Featurize, TrainTestSplit, LoadRawData
from utils.EvaluationsStub import Evaluation


def compare_roc_curves_by_selected_features(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, fs_configs, thresholds, config):

    graphs = []
    legends = []
    original_fpr_fnr = []
    for i, fs_config in enumerate(fs_configs):
        legends.append('Config {}'.format(fs_config['name']))
        del fs_config['name']

        (xTrain, xTest) = Featurize(xTrainRaw, xTestRaw, **fs_config)
        yTrain = yTrainRaw
        yTest = yTestRaw
        model = RandomForestModel(numTrees=config['num_trees'],
                                  bagging_w_replacement=config['bagging_w_replacement'])
        model.fit(xTrain, yTrain, min_to_split=config['min_to_split'])
        yTestPredicted_prob = model.predict_probabilities(xTest)

        cont_length_fpr_fnr = []
        for threshold in thresholds:
            yTestPredicted = [int(p >= threshold) for p in yTestPredicted_prob]
            # featured = [int(y < threshold) for y in yTestPredicted]
            ev = Evaluation(yTest, yTestPredicted)
            # if threshold == 0:
            #    assert (ev.fpr == 0.0), ev
            print(ev)
            cont_length_fpr_fnr.append((ev.fpr, ev.fnr))
        graphs.append(cont_length_fpr_fnr)

    # plotting
    start = thresholds[0]
    end = thresholds[-1]
    step = thresholds[1] - thresholds[0]
    fname = 'prob1{}_{}_{}_{}.png'.format(inspect.stack()[0][3], start, end, step)
    img_fname = os.path.join(report_path, fname)
    draw_accuracies(graphs,
                    'False Positive Rate', 'False Negative Rate', '',
                    img_fname,
                    legends=legends,
                    invert_yaxis=True,
                    data_pt='-')


if __name__ == '__main__':
    (xRaw, yRaw) = LoadRawData(kDataPath, includeLeftEye=True, includeRightEye=True)
    (xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit(xRaw, yRaw, percentTest=.25)

    config = {'min_to_split': 2,
              'bagging_w_replacement': True,
              'num_trees': 40,
              'feature_restriction': 100}
    configs = []
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

    fs_configs = [config0, config1, config2, config3]

    start = 0
    end = 1
    N = 100
    thresholds = [x / N for x in range(N + 1)]
    compare_roc_curves_by_selected_features(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw, fs_configs, thresholds, config)
