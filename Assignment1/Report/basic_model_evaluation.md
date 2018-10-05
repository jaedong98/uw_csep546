# Assignment#1 - Basic Model Evaluation

## Jae Dong Hwang

I implemented the evaluation methods as below in *EvaluationsStub.py*.

```python
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
            raise ValueError("Unexpected pair value {}".format(pair))

    return tp, fp, fn, tn


def ConfusionMatrix(y, yPredicted):

    tp, fp, fn, tn = _cm_calculator(y, yPredicted)
    w = 10

    header = '{}|{}|{}'.format(''.center(w), '1'.center(w), '0'.center(w))
    splitter = ''.center(w * 3 + 2, '-')
    yhat1 = '{}|{}|{}'.format('1'.center(w), '(TP) {}'.format(
        tp).center(w), '(FN) {}'.format(fn).center(w))
    yhat0 = '{}|{}|{}'.format('0'.center(w), '(FP) {}'.format(
        fp).center(w), '(TN) {}'.format(tn).center(w))
    return '\n'.join([header, splitter, yhat1, yhat0])
```

The output of *StartPoint1.py* with the implementation is:

```bash
Train is 0.130383 percent spam.
Test is 0.144907 percent spam.
<bound method MostCommonModel.predict of <MostCommonModel.MostCommonModel object at 0x00000228836E97B8>>
### 'Most Common' model
          |    1     |    0
--------------------------------
    1     |  (TP) 0  | (FN) 202
    0     |  (FP) 0  |(TN) 1192
Accuracy: 0.8550932568149211
Precision: 0.0
Recall: 0.0
FPR: 1.0
FNR: 0.0
### Heuristic model
          |    1     |    0
--------------------------------
    1     | (TP) 168 | (FN) 34
    0     | (FP) 83  |(TN) 1109
Accuracy: 0.9160688665710186
Precision: 0.6693227091633466
Recall: 0.8316831683168316
FPR: 0.16831683168316833
FNR: 0.06963087248322147
```