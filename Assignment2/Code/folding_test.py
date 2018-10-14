from k_fold_cross_validation import fold_data

xTrainRaw = ['a', 'b', 'c']
trainings, validations = fold_data(xTrainRaw, 3)
for t, v in zip(trainings, validations):
    print("Training: {} Validation: {}".format(t, v))