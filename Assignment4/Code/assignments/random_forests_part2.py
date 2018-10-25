import os
import random
import time
from Assignment4.Code import report_path
from assignments.random_forests_part1 import calculate_accuracies
from utils.Assignment4Support import draw_random_forest_accuracy_variances

configs = [{'min_to_split': 2, 'use_bagging': True, 'feature_restriction': 20},
           {'min_to_split': 50, 'use_bagging': True, 'feature_restriction': 20},
           {'min_to_split': 2, 'use_bagging': False, 'feature_restriction': 20},
           {'min_to_split': 2, 'use_bagging': True, 'feature_restriction': 0}]


def generate_comparision_accuracies(numTrees_options=[1, 20, 40, 60, 80],
                                    configs=configs,
                                    seed=0):

    accuracies_per_config = []
    for config in configs:
        print("Configuration: {}".format(config))
        accuracies = []
        for numTrees in numTrees_options:
            config['numTrees'] = numTrees
            config['seed'] = seed
            accuracy = calculate_accuracies(**config)[0]
            accuracies.append((numTrees, accuracy))
        accuracies_per_config.append(accuracies)

    return accuracies_per_config


def report(numTrees_options=[1, 20, 40, 60, 80], configs=configs, seed=0):
    print("Random seed: {}".format(seed))
    accuracies = generate_comparision_accuracies(numTrees_options, configs, seed)
    xlabel = 'numTrees'
    ylabel = 'Accuracies'
    title = 'Random Forests Accuracy Comparison'
    fname = 'prob2_part2_accuracy_cmp_{}_randseed_{}.png'\
        .format('_'.join([str(n) for n in numTrees_options]), seed)
    img_fname = os.path.join(report_path, fname)
    legends = ['Config {}'.format(i) for i in range(len(configs))]

    draw_random_forest_accuracy_variances(accuracies,
                                          xlabel, ylabel,
                                          title,
                                          img_fname,
                                          legends)


if __name__ == '__main__':

    tic = time.time()
    report(numTrees_options=[1, 20, 40, 60, 80], seed=0)
    print('{} mins'.format((time.time() - tic) / 60.))

