import os

from Assignment4.Code import report_path
from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


def calculate_accuracies(numTrees=10,
                         bagging_w_replacement=True,
                         feature_restriction=0,
                         min_to_split=2,
                         seed=0,
                         with_noise=True):
    rfm = RandomForestModel(numTrees=numTrees,
                            bagging_w_replacement=bagging_w_replacement,
                            feature_restriction=feature_restriction,
                            seed=seed)
    xTrain, xTest, yTrain, yTest = get_featurized_xs_ys(with_noise)
    rfm.fit(xTrain, yTrain, min_to_split=min_to_split)
    yTestPredicted = rfm.predict(xTest)
    ev = Evaluation(yTest, yTestPredicted)
    print("Overall prediction:")
    print(ev)
    accuracies = [ev.accuracy]
    for prediction in rfm.predictions:
        ev = Evaluation(yTest, prediction)
        accuracies.append(ev.accuracy)

    return accuracies


def create_accuracy_comparison_tables(numTrees=10,
                                      bagging_w_replacement=True,
                                      feature_restriction=0,
                                      min_to_split=2,
                                      seed=10,
                                      w=25,
                                      with_noise=True):

    accuracies = calculate_accuracies(numTrees,
                                      bagging_w_replacement,
                                      feature_restriction,
                                      min_to_split,
                                      seed,
                                      with_noise)
    table = '|'
    table += str('{}|' * 2).format(*[s.center(w) for s in ["Trees", "Accuracies"]])
    tree_names = ['Full'.center(w)] + ['Tree {}'.center(w).format(i) for i in range(numTrees)]
    table += '\n|'
    table += str('-' * (w) + '|') * 2
    for name, accu in zip(tree_names, accuracies):
        table += '\n|'
        table += str('{}|' * 2).format(*[str(s).center(w) for s in [name, accu]])

    print(table)

    fname = 'prob1_part1_tree_accurarices_{}_b{},r{}.md'\
        .format(numTrees, bagging_w_replacement, feature_restriction)
    md = os.path.join(report_path, fname)

    with open(md, 'w') as f:
        f.write(table)
        f.write('\n')
        f.write('\nUse Bagging: {}'.format(bagging_w_replacement))
        f.write('\nFeature Restriction: {}'.format(feature_restriction))
        f.write('\nMinToSplit: {}'.format(min_to_split))
        f.write('\nSeed for random: {}'.format(seed))

    print("Generated report: {}".format(md))
    return accuracies[0]


if __name__ == '__main__':

    config = -1

    if config == -1:  # basic configuration
        create_accuracy_comparison_tables(numTrees=10,
                                          bagging_w_replacement=True,
                                          feature_restriction=20,
                                          min_to_split=2,
                                          seed=10,
                                          with_noise=True)

    if config == 2:
        create_accuracy_comparison_tables(numTrees=20,
                                          bagging_w_replacement=True,
                                          feature_restriction=20,
                                          min_to_split=2,
                                          seed=10000,
                                          with_noise=True)
