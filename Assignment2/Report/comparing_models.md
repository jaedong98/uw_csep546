# Assignment#2. Comparing Models

## Jae Dong Hwang

Implement code to estimate the 95% range for your accuracy estimates. In the future always include 95% confidence ranges whenever you turn in accuracy estimates.

Recall:
Upper = Accuracy + 1.96 * sqrt((Accuracy * (1 - Accuracy) / n))
Lower = Accuracy - 1.96 * sqrt((Accuracy * (1 - Accuracy) / n))

Evaluate two settings from your feature selection assignment: 10 words selected by mutual information and 10 words selected by frequency).

Implement cross validation that supports any number of folds from 2 to n. Verify that it is selecting the correct data into each fold. Be prepared to run all current evaluations on the result (precision, recall, false positive rate, false negative rate).

Run a 5 fold crossvalidation evaluation of tw models from the previous assignment (10 words selected with mutual info, 10 words selected with frequency).

Run this ONLY on the previous training data (hold out the previous test data).

Hand in a clearly labeled table with:

## The accuracy estimates from the train/test split run with error bounds

* Accuracy Estimates w/ Zn=1.96
  |      Feature Selections      |           Accuracy           |            Upper             |            Lower             |
  |------------------------------|------------------------------|------------------------------|------------------------------|
  |       Top 10 Frequency       |      0.8550932568149211      |      0.8657645971081168      |      0.8444219165217254      |
  |          Top 10 MI           |      0.9239598278335724      |      0.9319953854766465      |      0.9159242701904984      |

## The accuracy estimates from the cross validations runs for the two model variants with error bound

* Accuracy Estimate from Cross Validation

  |      Feature Selections      |  TotalCorrect |       N       |           Accuracy           |            Upper             |            Lower             |
  |------------------------------|---------------|---------------|------------------------------|------------------------------|------------------------------|
  |       Top 10 Frequency       |      3294     |       10      |      0.7880382775119618      |      0.8004282490023151      |      0.7756483060216084      |
  |          Top 10 MI           |      3891     |       10      |      0.9308612440191387      |      0.9385520381370541      |      0.9231704499012233      |

***
