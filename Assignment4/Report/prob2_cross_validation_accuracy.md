
Overall Accuracy: 0.8913875598086124
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.8913875598086124      |      0.9008203679327798      |      0.881954751684445       |
|        Hold-out Data         |      0.896700143472023       |      0.9059267374420649      |      0.8874735495019811      |
Configuration:
 * name: Improved
 * iterations: 10000
 * min_to_stop: 2
 * bagging_w_replacement: True
 * num_trees: 40
 * feature_restriction: 50
 * feature_selection_by_mi: 250
 * feature_selection_by_frequency: 0
 * include_handcrafted_features: True

DecisionTreeModel for 0th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 86  | (FN) 77  |
|    0     |  (FP) 4  | (TN) 669 |
Accuracy: 0.90311004784689
Precision: 0.9555555555555556
Recall: 0.5276073619631901
FPR: 0.005943536404160475
FNR: 0.4723926380368098


DecisionTreeModel for 1th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 75  | (FN) 79  |
|    0     | (FP) 16  | (TN) 666 |
Accuracy: 0.8863636363636364
Precision: 0.8241758241758241
Recall: 0.487012987012987
FPR: 0.02346041055718475
FNR: 0.512987012987013


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 73  | (FN) 77  |
|    0     | (FP) 12  | (TN) 674 |
Accuracy: 0.8935406698564593
Precision: 0.8588235294117647
Recall: 0.4866666666666667
FPR: 0.01749271137026239
FNR: 0.5133333333333333


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 81  | (FN) 105 |
|    0     |  (FP) 7  | (TN) 643 |
Accuracy: 0.8660287081339713
Precision: 0.9204545454545454
Recall: 0.43548387096774194
FPR: 0.010769230769230769
FNR: 0.5645161290322581


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 87  | (FN) 67  |
|    0     | (FP) 10  | (TN) 672 |
Accuracy: 0.9078947368421053
Precision: 0.8969072164948454
Recall: 0.564935064935065
FPR: 0.01466275659824047
FNR: 0.43506493506493504
