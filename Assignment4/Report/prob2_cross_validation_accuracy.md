
Overall Accuracy: 0.8937799043062201
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.8937799043062201      |      0.9031207579520755      |      0.8844390506603648      |
|        Hold-out Data         |      0.8988522238163558      |      0.9079931509821646      |      0.8897112966505469      |
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
|    1     | (TP) 93  | (FN) 70  |
|    0     |  (FP) 6  | (TN) 667 |
Accuracy: 0.9090909090909091
Precision: 0.9393939393939394
Recall: 0.5705521472392638
FPR: 0.008915304606240713
FNR: 0.4294478527607362


DecisionTreeModel for 1th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 79  | (FN) 75  |
|    0     | (FP) 17  | (TN) 665 |
Accuracy: 0.8899521531100478
Precision: 0.8229166666666666
Recall: 0.512987012987013
FPR: 0.024926686217008796
FNR: 0.487012987012987


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 77  | (FN) 73  |
|    0     | (FP) 14  | (TN) 672 |
Accuracy: 0.895933014354067
Precision: 0.8461538461538461
Recall: 0.5133333333333333
FPR: 0.02040816326530612
FNR: 0.4866666666666667


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 84  | (FN) 102 |
|    0     |  (FP) 8  | (TN) 642 |
Accuracy: 0.868421052631579
Precision: 0.9130434782608695
Recall: 0.45161290322580644
FPR: 0.012307692307692308
FNR: 0.5483870967741935


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 87  | (FN) 67  |
|    0     | (FP) 12  | (TN) 670 |
Accuracy: 0.9055023923444976
Precision: 0.8787878787878788
Recall: 0.564935064935065
FPR: 0.017595307917888565
FNR: 0.43506493506493504
