# Homework4. Problem 1 Random Forests

## Jae Dong Hwang

### Build Random Forest Model

#### Build a model with numTrees = 10, minSplit 2, use Bagging and FeatureRestriction. Create a table that has accuracies of the 10 individual trees, along with the accuracy of the full random forest (after the individual trees vote) on xTest.

|          Trees          |        Accuracies       |
|-------------------------|-------------------------|
|           Full          |    0.8041606886657102   |
|          Tree 0         |    0.639167862266858    |
|          Tree 1         |    0.7826398852223816   |
|          Tree 2         |    0.7360114777618364   |
|          Tree 3         |    0.7883787661406025   |
|          Tree 4         |    0.7962697274031564   |
|          Tree 5         |    0.7948350071736011   |
|          Tree 6         |    0.8106169296987088   |
|          Tree 7         |    0.8263988522238164   |
|          Tree 8         |    0.8041606886657102   |
|          Tree 9         |    0.8041606886657102   |

Use Bagging: True
Feature Restriction: 20
MinToSplit: 2
Seed for random: 10


#### Run parameter sweeps for numTrees in: [1, 20, 40, 60, 80], for each of these settings:

* Config 0 - minSplit = 2, Bagging, and FeatureRestriction = 20
* Config 1 - minSplit of 50, Bagging, and FeatureRestriction = 20
* Config 2 - minSplit = 2, NO Bagging, and FeatureRestriction = 20
* Config 3 - minSplit = 2, Bagging, and NO FeatureRestriction = 0

##### Produce a plot with numTrees on x-axis and the hold-out set accuracy for each of these variations on the y-axis (that means 4 lines).

![prob1_part2_accuracy_cmp_[1, 20, 40, 60, 80]_randseed_0](prob1_part2_accuracy_cmp_1_20_40_60_80_randseed_0.png)

