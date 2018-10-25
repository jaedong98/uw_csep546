from model.RandomForestsModel import RandomForestModel
from utils.EvaluationsStub import Evaluation
from utils.data_loader import get_featurized_xs_ys


def create_accuracy_comparison_tables(numTrees=10,
                                      use_bagging=True,
                                      feature_restriction=0,
                                      min_to_split=2,
                                      w=25):
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
    table += str('{}|' * 2).format(*[s.center(w) for s in ["Trees", "Accuracies"]])
    tree_names = ['Full'.center(w)] + ['Tree {}'.center(w).format(i) for i in range(numTrees)]
    table += '\n|'
    table += str('-' * (w) + '|') * 2
    for name, accu in zip(tree_names, accuracies):
        table += '\n|'
        table += str('{}|' * 2).format(*[str(s).center(w) for s in [name, accu]])

    print(table)


if __name__ == '__main__':

    create_accuracy_comparison_tables(2)