
Overall Accuracy: 0.8942583732057416
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.8942583732057416      |      0.9035806594052523      |      0.8849360870062309      |
|        Hold-out Data         |      0.8995695839311334      |      0.9086816727179909      |      0.8904574951442759      |
Configuration:
 * name: Improved
 * iterations: 10000
 * min_to_stop: 2
 * bagging_w_replacement: True
 * num_trees: 40
 * feature_restriction: 100
 * feature_selection_by_mi: 250
 * feature_selection_by_frequency: 0
 * include_handcrafted_features: True

DecisionTreeModel for 0th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 92  | (FN) 71  |
|    0     |  (FP) 6  | (TN) 667 |
Accuracy: 0.9078947368421053
Precision: 0.9387755102040817
Recall: 0.5644171779141104
FPR: 0.008915304606240713
FNR: 0.43558282208588955


DecisionTreeModel for 1th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 78  | (FN) 76  |
|    0     | (FP) 15  | (TN) 667 |
Accuracy: 0.8911483253588517
Precision: 0.8387096774193549
Recall: 0.5064935064935064
FPR: 0.021994134897360705
FNR: 0.4935064935064935


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 76  | (FN) 74  |
|    0     | (FP) 11  | (TN) 675 |
Accuracy: 0.8983253588516746
Precision: 0.8735632183908046
Recall: 0.5066666666666667
FPR: 0.016034985422740525
FNR: 0.49333333333333335


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 86  | (FN) 100 |
|    0     |  (FP) 7  | (TN) 643 |
Accuracy: 0.8720095693779905
Precision: 0.9247311827956989
Recall: 0.46236559139784944
FPR: 0.010769230769230769
FNR: 0.5376344086021505


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 87  | (FN) 67  |
|    0     | (FP) 15  | (TN) 667 |
Accuracy: 0.9019138755980861
Precision: 0.8529411764705882
Recall: 0.564935064935065
FPR: 0.021994134897360705
FNR: 0.43506493506493504
