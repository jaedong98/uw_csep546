# Assignment#2. Feature Engineering

## Jae Dong Hwang

**Add bag of words features to your spam domain solution**

Support frequency based feature selection, top N
Support mutual information based features selection top N
Tokenize in the simplest way possible (by splitting on whitespace)

Recall:

MutiualInformation(X,Y) = Sum over every value X has and Y has:

            P(has X, has Y) * log_2 P(has X, has Y) / (P(has X) * P(has Y))

And use smoothing when calculating the probabilities:

P(*) = (# observed + 1) / (total samples + 2)

HAND IN:

A document that contains the following tables (clearly labeled!)

* A table showing the accuracy with each one left out, compared to a model built with all of them.

  | Leave-out-Features |      Accuracy      |
  |--------------------|--------------------|
  |   w/o IS_LONGGER   | 0.8895265423242468 |
  |   w/o HAS_NUMBER   | 0.857245337159254  |
  |  w/o CONTAIN_CALL  | 0.8938307030129125 |
  |   w/o CONTAIN_TO   | 0.8974175035868006 |
  |  w/o CONTAIN_YOUR  | 0.9167862266857962 |
  | w/ All of Features | 0.8931133428981348 |

* A list of the top 10 bag of word features selected by filtering by frequency.
  
  | Features | Frequency |
  | :------: | :-------: |
  | to       | 1586      |
  | you      | 1174      |
  | I        | 1099      |
  | a        | 993       |
  | the      | 890       |
  | and      | 629       |
  | is       | 592       |
  | in       | 586       |
  | i        | 558       |
  | u        | 542       |

* A list of the top 10 bag of word features selected by filtering by mutual information.

  | Features | Mutual Information    |
  | -------- | --------------------- |
  | I        | 0.0037431778649602105 |
  | Call     | 0.0025203798599061087 |
  | i        | 0.002356358770874558  |
  | FREE     | 0.0022381150276038434 |
  | claim    | 0.0019205695585422415 |
  | &        | 0.0018018071141155463 |
  | my       | 0.0016577371208125879 |
  | mobile   | 0.00164062330676438   |
  | To       | 0.0016346171629362423 |
  | Txt      | 0.0015661937627271662 |

2 points --

0.5 point -- Run gradient descent to 50,000 iterations with the top 10 words by frequency.
0.5 point -- Run gradient descent to 50,000 iterations with the top 10 words by mutual information.
0.5 point -- Run gradient descent to 50,000 iterations with the better of these PLUS the hand crafted features from the framework.
0.5 point -- Run gradient descent to 50,000 iterations of the previous setting with 100 words plus hand-crafted instead of 10.
Hand in a clearly labeled table comparing the accuracies of these methods

***

```python

for f in featureSelectionMethodsToTry:
    (trainX, trainY, fParameters) = FeaturizeTraining(rawTrainX, rawTrainY, f)
    (validationX, validationY) = FeaturizeValidation(rawValidationX, rawValidationY, f, fParameters)

for p in parametersToTry:
    model.fit(trainX, trainY, p)
    accuracies[p, f] = evaluate(validationY, model.predict(validationX))

(bestPFound, bestFFound) = bestSettingFound(accuracies)

(finalTrainX, finalTrainY, fParameters) =
    FeaturizeTraining(rawTrainX + rawValidationX, rawTrainY + rawValidationY, bestFFound)

(testX, testY) = FeaturizeValidation(rawTextX, rawTestY, bestFFound, fParameters)

finalModel.fit(finalTrainX, finalTrainY, bestPFound)

estimateOfGeneralizationPerformance = evaluate(testY, model.predict(testX))

```