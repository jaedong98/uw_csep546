from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


def create_accuracy_comparison_tables(numTrees=10,
                                      use_bagging=True,
                                      feature_restriction=0,
                                      min_to_split=2,
                                      w=20):
    rfm = RandomForestModel(numTrees=numTrees,
                            use_bagging=use_bagging,
                            feature_restriction=feature_restriction,
                            seed=0)
    xTrain, xTest, yTrain, yTest = get_featurized_xs_ys()
    rfm.fit(xTrain, yTrain, min_to_split=min_to_split)
    yTestPredicted = rfm.predict(xTest)
    accuracies = [Evaluation(yTest, yTestPredicted).accuracy]
    for prediction in rfm.predictions:
        accuracies.append(Evaluation(yTest, prediction).accuracy)

    table = '|'
    headers = [' '.center(w), 'full'.center(w)] + ['Tree {}'.center(w).format(i) for i in range(numTrees)]
    table += str('{}|' * (numTrees + 2)).format(*headers)
    table += '\n|'
    table += str('-' * (w) + '|') * (numTrees + 2)
    table += '\n|'
    row = ["Accuracies".center(w)] + ['{}'.format(i).center(w) for i in accuracies]
    table += str('{}|' * (numTrees + 2)).format(*row)

    print(table)


if __name__ == '__main__':

    create_accuracy_comparison_tables(2)