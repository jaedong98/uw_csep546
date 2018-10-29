
Overall Accuracy: 0.8854066985645933
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.8854066985645933      |      0.8950631808460365      |      0.8757502162831501      |
|        Hold-out Data         |      0.8916786226685797      |      0.9011003210000925      |      0.8822569243370668      |
Configuration:
 * name: Improved
 * iterations: 10000
 * min_to_stop: 2
 * bagging_w_replacement: True
 * num_trees: 40
 * feature_restriction: 100
 * feature_selection_by_mi: 100
 * feature_selection_by_frequency: 0
 * include_handcrafted_features: True

DecisionTreeModel for 0th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 91  | (FN) 72  |
|    0     | (FP) 18  | (TN) 655 |
Accuracy: 0.8923444976076556
Precision: 0.8348623853211009
Recall: 0.558282208588957
FPR: 0.02674591381872214
FNR: 0.44171779141104295


DecisionTreeModel for 1th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 76  | (FN) 78  |
|    0     | (FP) 26  | (TN) 656 |
Accuracy: 0.8755980861244019
Precision: 0.7450980392156863
Recall: 0.4935064935064935
FPR: 0.03812316715542522
FNR: 0.5064935064935064


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 81  | (FN) 69  |
|    0     | (FP) 25  | (TN) 661 |
Accuracy: 0.8875598086124402
Precision: 0.7641509433962265
Recall: 0.54
FPR: 0.03644314868804665
FNR: 0.46


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 97  | (FN) 89  |
|    0     | (FP) 21  | (TN) 629 |
Accuracy: 0.868421052631579
Precision: 0.8220338983050848
Recall: 0.521505376344086
FPR: 0.03230769230769231
FNR: 0.478494623655914


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 96  | (FN) 58  |
|    0     | (FP) 23  | (TN) 659 |
Accuracy: 0.90311004784689
Precision: 0.8067226890756303
Recall: 0.6233766233766234
FPR: 0.03372434017595308
FNR: 0.37662337662337664
