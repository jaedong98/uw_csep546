* Accuracies and Parameters Selected
  * 3x3 grid + y gradients
    * Best Accuracy: 0.8193069306930693 (Param Sweep with numTrees ([20, 40, 60, 80]))
    * min_to_split: 20
    * bagging_w_replacement: True
    * num_trees: 60
    * feature_restriction: 100
      * Statistics: 

      |          |    1     |    0     |
      |----------|----------|----------|
      |    1     | (TP) 482 | (FN) 116 |
      |    0     | (FP) 103 | (TN) 511 |
      Accuracy: 0.8193069306930693
      Precision: 0.8239316239316239
      Recall: 0.8060200668896321
      FPR: 0.16775244299674266
      FNR: 0.1939799331103679
  * 3x3 grid + x gradients
    * Best Accuracy: 0.8448844884488449 (Param Sweep with numTrees ([20, 40, 60, 80]))
    * min_to_split: 20
    * bagging_w_replacement: True
    * num_trees: 40
    * feature_restriction: 100
      * Statistics: 

      |          |    1     |    0     |
      |----------|----------|----------|
      |    1     | (TP) 498 | (FN) 100 |
      |    0     | (FP) 88  | (TN) 526 |
      Accuracy: 0.8448844884488449
      Precision: 0.8498293515358362
      Recall: 0.8327759197324415
      FPR: 0.14332247557003258
      FNR: 0.16722408026755853
  * Histogram of a image y-gradients
    * Best Accuracy: 0.8061056105610561 (Param Sweep with numTrees ([20, 40, 60, 80]))
    * min_to_split: 20
    * bagging_w_replacement: True
    * num_trees: 60
    * feature_restriction: 100
      * Statistics: 

      |          |    1     |    0     |
      |----------|----------|----------|
      |    1     | (TP) 447 | (FN) 151 |
      |    0     | (FP) 84  | (TN) 530 |
      Accuracy: 0.8061056105610561
      Precision: 0.8418079096045198
      Recall: 0.7474916387959866
      FPR: 0.13680781758957655
      FNR: 0.2525083612040134
  * Histogram of a image y-gradients
    * Best Accuracy: 0.7904290429042904 (Param Sweep with numTrees ([20, 40, 60, 80]))
    * min_to_split: 20
    * bagging_w_replacement: True
    * num_trees: 80
    * feature_restriction: 100
      * Statistics: 

      |          |    1     |    0     |
      |----------|----------|----------|
      |    1     | (TP) 427 | (FN) 171 |
      |    0     | (FP) 83  | (TN) 531 |
      Accuracy: 0.7904290429042904
      Precision: 0.8372549019607843
      Recall: 0.7140468227424749
      FPR: 0.13517915309446255
      FNR: 0.28595317725752506