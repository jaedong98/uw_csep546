

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning(
            "Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")


def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)


def Precision(y, yPredicted):
    tp, fp, _, _ = _cm_calculator(y, yPredicted)
    if tp == 0:
        return 0.0
    return tp / (tp + fp)


def Recall(y, yPredicted):
    tp, _, fn, _ = _cm_calculator(y, yPredicted)
    if tp == 0:
        return 0.0
    return tp / (tp + fn)


def FalseNegativeRate(y, yPredicted):
    tp, _, fn, n = _cm_calculator(y, yPredicted)
    if fn == 0:
        return 0.0
    return fn / (tp + fn)


def FalsePositiveRate(y, yPredicted):
    _, fp, _, tn = _cm_calculator(y, yPredicted)
    if fp == 0:
        return 0.0
    return fp / (fp + tn)


def _cm_calculator(y, yPredicted):
    tp, fp, fn, tn = 0, 0, 0, 0
    for y, yhat in zip(y, yPredicted):
        y_yhat = (y, yhat)
        if y_yhat == (1, 1):
            tp += 1
        elif y_yhat == (1, 0):
            fn += 1
        elif y_yhat == (0, 1):
            fp += 1
        elif y_yhat == (0, 0):
            tn += 1
        else:
            raise ValueError("Unexpected pair value {}".format(y_yhat))

    return tp, fp, fn, tn


def ConfusionMatrix(y, yPredicted, indent=''):

    tp, fp, fn, tn = _cm_calculator(y, yPredicted)
    w = 10

    header = '{}|{}|{}|{}|'.format(indent, ''.center(w), '1'.center(w), '0'.center(w))

    splitter = '{}|'.format(indent) + '-' * 10
    splitter += '|' + '-' * 10
    splitter += '|' + '-' * 10
    splitter += '|'
    yhat1 = '{}|{}|{}|{}|'.format(indent, '1'.center(w), '(TP) {}'.format(
        tp).center(w), '(FN) {}'.format(fn).center(w))
    yhat0 = '{}|{}|{}|{}|'.format(indent, '0'.center(w), '(FP) {}'.format(
        fp).center(w), '(TN) {}'.format(tn).center(w))
    return '\n'.join([header, splitter, yhat1, yhat0])


def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))


def EvaluateAll(y, yPredicted, indent=''):

    results = "{}* Statistics: ".format(indent)
    results += "\n"
    results += "\n{}".format(ConfusionMatrix(y, yPredicted, indent))
    results += "\n{}Accuracy: {}".format(indent, Accuracy(y, yPredicted))
    results += "\n{}Precision: {}".format(indent, Precision(y, yPredicted))
    results += "\n{}Recall: {}".format(indent, Recall(y, yPredicted))
    results += "\n{}FPR: {}".format(indent, FalsePositiveRate(y, yPredicted))
    results += "\n{}FNR: {}".format(indent, FalseNegativeRate(y, yPredicted))

    return results


class Evaluation(object):

    def __init__(self, y, yPredicted, indent=''):
        self.y = y
        self.yPredicted = yPredicted
        self.accuracy = Accuracy(y, yPredicted)
        self.precision = Precision(y, yPredicted)
        self.recall = Recall(y, yPredicted)
        self.fpr = FalsePositiveRate(y, yPredicted)
        self.fnr = FalseNegativeRate(y, yPredicted)
        self.indent = indent

    def __repr__(self):
        return EvaluateAll(self.y, self.yPredicted, self.indent)