
Overall Accuracy: 0.9822966507177033
Accuracy Estimate Comparison: 
|            Cases             |           Accuracy           |            Upper             |            Lower             |
|------------------------------|------------------------------|------------------------------|------------------------------|
|        Training Data         |      0.9822966507177033      |      0.9862944131366549      |      0.9782988882987518      |
|        Hold-out Data         |      0.9813486370157819      |      0.9854500630081059      |      0.9772472110234579      |
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
|    1     | (TP) 98  |  (FN) 9  |
|    0     |  (FP) 0  | (TN) 729 |
Accuracy: 0.9892344497607656
Precision: 1.0
Recall: 0.9158878504672897
FPR: 0.0
FNR: 0.08411214953271028


DecisionTreeModel for 1th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 93  | (FN) 15  |
|    0     |  (FP) 0  | (TN) 728 |
Accuracy: 0.9820574162679426
Precision: 1.0
Recall: 0.8611111111111112
FPR: 0.0
FNR: 0.1388888888888889


DecisionTreeModel for 2th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 91  | (FN) 11  |
|    0     |  (FP) 5  | (TN) 729 |
Accuracy: 0.9808612440191388
Precision: 0.9479166666666666
Recall: 0.8921568627450981
FPR: 0.006811989100817439
FNR: 0.10784313725490197


DecisionTreeModel for 3th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 100 | (FN) 18  |
|    0     |  (FP) 2  | (TN) 716 |
Accuracy: 0.9760765550239234
Precision: 0.9803921568627451
Recall: 0.847457627118644
FPR: 0.002785515320334262
FNR: 0.15254237288135594


DecisionTreeModel for 4th folding
* Statistics: 

|          |    1     |    0     |
|----------|----------|----------|
|    1     | (TP) 100 | (FN) 10  |
|    0     |  (FP) 4  | (TN) 722 |
Accuracy: 0.9832535885167464
Precision: 0.9615384615384616
Recall: 0.9090909090909091
FPR: 0.005509641873278237
FNR: 0.09090909090909091
