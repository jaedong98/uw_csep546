# Assignment#2. Categorize Mistakes

## Jae Dong Hwang

**Implement a way to get the raw context for the samples where your model is most-wrong. Recall this includes examples where the true answer was 1, but the model gives very low probabilities, and examples where the true answer was 0, but gives very high probabilities.**

* **Produce a list of the 20 worst false positives made by running logistic regression on the initial train/test split with 10 mutual information features**
  ### UPDATE !!(Running by categorize_mistake.py -> category_mistake_false_positives)

* **Produce a list of the 20 worst false negatives made by running logistic regression on the initial train/test split with 10 mutual information features**
  ### UPDATE !! (Running by categorize_mistake.py -> category_mistake_false_negatives)

* **Categorize the false positives into at least 4 categories.**

* **Categorize the false negatives into at least 4 categories.**

* **In no more than 150 words describe the insight you got from this process, including one new heuristic feature you think would reduce the bad false positives, and one that would reduce the bad false negatives.**

  To reduce false positive:
  * Preprocessing the train the objects that triggered the false positive as negative sms message.

  To reduce false negative:
  * If possible, increasing similarity between training and testing would decrease the false negative rate.
  * Increasing the volume of training data would reduce false negatives.
  
***
