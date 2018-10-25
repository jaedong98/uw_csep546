import os
from Assignment4.Code import report_path
from assignments.randome_forests_part1 import calculate_accuracies
from utils.Assignment4Support import draw_random_forest_accuracy_variances

configs = [{'min_to_split': 2, 'use_bagging': True, 'feature_restriction': 20},
           {'min_to_split': 50, 'use_bagging': True, 'feature_restriction': 20},
           {'min_to_split': 2, 'use_bagging': False, 'feature_restriction': 20},
           {'min_to_split': 2, 'use_bagging': True, 'feature_restriction': 0}]


def generate_comparision_accuracies(numTrees_options=[1, 20, 40, 60, 80],
                                    configs=configs):


    accuracies_per_config = []
    for config in configs:
        accuracies = []
        for numTrees in numTrees_options:
            config['numTrees'] = numTrees
            accuracy = calculate_accuracies(**config)[0]
            accuracies.append((numTrees, accuracy))
        accuracies_per_config.append(accuracies)

    return accuracies_per_config


def report(numTrees_options=[1, 20, 40, 60, 80], configs=configs):

    accuracies = generate_comparision_accuracies(numTrees_options, configs)
    xlabel = 'numTrees'
    ylabel = 'Accuracies'
    title = 'Random Forests Accuracy Comparison'
    fname = 'prob1_part2_accuracy_cmp_{}.png'.format(numTrees_options)
    img_fname = os.path.join(report_path, fname)
    legends = ['Config {}'.format(i) for i in range(len(configs))]

    draw_random_forest_accuracy_variances(accuracies,
                                          xlabel, ylabel,
                                          title,
                                          img_fname,
                                          legends)


if __name__ == '__main__':
    report(numTrees_options=[1, 20, 40, 60, 80])
