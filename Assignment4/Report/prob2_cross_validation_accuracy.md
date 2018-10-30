
Overall Accuracy: 0.8952153110047847
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.8952153110047847      |      0.9045002829432074      |      0.885930339066362       |
|        Hold-out Data         |      0.9031563845050216      |      0.912122099020048       |      0.8941906699899951      |
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
|    0     | (FP) 14  | (TN) 668 |
Accuracy: 0.8923444976076556
Precision: 0.8478260869565217
Recall: 0.5064935064935064
FPR: 0.020527859237536656
FNR: 0.4935064935064935


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 77  | (FN) 73  |
|    0     | (FP) 11  | (TN) 675 |
Accuracy: 0.8995215311004785
Precision: 0.875
Recall: 0.5133333333333333
FPR: 0.016034985422740525
FNR: 0.4866666666666667


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 88  | (FN) 98  |
|    0     |  (FP) 7  | (TN) 643 |
Accuracy: 0.8744019138755981
Precision: 0.9263157894736842
Recall: 0.4731182795698925
FPR: 0.010769230769230769
FNR: 0.5268817204301075


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 86  | (FN) 68  |
|    0     | (FP) 14  | (TN) 668 |
Accuracy: 0.9019138755980861
Precision: 0.86
Recall: 0.5584415584415584
FPR: 0.020527859237536656
FNR: 0.44155844155844154
