# Homework4. Problem 1 Random Forests

## Jae Dong Hwang

### Build Random Forest Model

#### Build a model with numTrees = 10, minSplit 2, use Bagging and FeatureRestriction=20. Create a table that has accuracies of the 10 individual trees, along with the accuracy of the full random forest (after the individual trees vote) on xTest.

|          Trees          |        Accuracies       |
|-------------------------|-------------------------|
|           Full          |    0.8041606886657102   |
|          Tree 0         |    0.8041606886657102   |
|          Tree 1         |    0.8256814921090387   |
|          Tree 2         |    0.6915351506456241   |
|          Tree 3         |    0.8041606886657102   |
|          Tree 4         |    0.7948350071736011   |
|          Tree 5         |    0.3629842180774749   |
|          Tree 6         |    0.6814921090387375   |
|          Tree 7         |    0.8091822094691535   |
|          Tree 8         |    0.7883787661406025   |
|          Tree 9         |    0.7173601147776184   |

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

![prob1_part2_accuracy_cmp_[1, 20, 40, 60, 80]_randseed_10000](prob1_part2_accuracy_cmp_1_20_40_60_80_randseed_10000.png)

