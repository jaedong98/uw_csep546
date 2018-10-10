# Assignment#2. Categorize Mistakes

## Jae Dong Hwang

Implement a way to get the raw context for the samples where your model is most-wrong. Recall this includes examples where the true answer was 1, but the model gives very low probabilities, and examples where the true answer was 0, but gives very high probabilities.

To do this you will need to have a version of model.fit that returns raw probabilities (without using a threshold).

HAND IN:

0.5 point: Produce a list of the 20 worst false positives made by running logistic regression on the initial train/test split with 10 mutual information features

0.5 points: Produce a list of the 20 worst false negatives made by running logistic regression on the initial train/test split with 10 mutual information features

Look at these mistakes and categories them:

0.5 points: categorize the false positives into at least 4 categories.

0.5 points: categorize the false negatives into at least 4 categories.

1 point: in no more than 150 words describe the insight you got from this process, including one new heuristic feature you think would reduce the bad false positives, and one that would reduce the bad false negatives.

***