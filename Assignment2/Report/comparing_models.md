# Assignment#2. Comparing Models

## Jae Dong Hwang

Implement code to estimate the 95% range for your accuracy estimates. In the future always include 95% confidence ranges whenever you turn in accuracy estimates.

Recall:
Upper = Accuracy + 1.96 * sqrt((Accuracy * (1 - Accuracy) / n))
Lower = Accuracy - 1.96 * sqrt((Accuracy * (1 - Accuracy) / n))

Evaluate two settings from your feature selection assignment: 10 words selected by mutual information and 10 words selected by frequency).

Implement cross validation that supports any number of folds from 2 to n. Verify that it is selecting the correct data into each fold. Be prepared to run all current evaluations on the result (precision, recall, false positive rate, false negative rate).

Run a 5 fold crossvalidation evaluation of tw models from the previous assignment (10 words selected with mutual info, 10 words selected with frequency).

Run this ONLY on the previous training data (hold out the previous test data).

Hand in a clearly labeled table with:

* 1 point -- the accuracy estimates from the train/test split run with error bounds

* 1 point --  the accuracy estimates from the cross validations runs for the two model variants with error bound

NOTE: if your gradient descent is slow (like mine is) these runs are going to start to take a long time. One possibility is to just-not-care — let it run over night or whatever. Another easy approach is to do several runs in parallel. You can do this manually or programmatically.

You could also choose to optimize your code, but don’t go overboard. That’s not the point of this assignment. In practice you should use an existing, highly-optimized implementation.

***
