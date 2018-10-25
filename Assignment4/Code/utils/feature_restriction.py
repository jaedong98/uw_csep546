import random


def restrict_features(data, selected_indices):
    """
    Restircts features by selecting features in each sample.

    :param data: xTrain or xTest (a list of samples(feature lists))
    :param selected_indices: N randomly selected feature index.
    :return: a list of 'selected' feature lists
    """
    if len(selected_indices) > len(data[0]):
        raise ValueError("Got indices more than features")

    if max(selected_indices) > (len(data[0]) - 1):
        raise ValueError("Maximum index({}) is outside range."
                         .format(max(selected_indices)))

    selected_data = []
    for features in data:
        n_features = []
        for i in selected_indices:
            n_features.append(features[i])
        selected_data.append(n_features)

    if not len(selected_data) == len(data):
        raise AssertionError("Missing data in output.")

    return selected_data


def select_random_indices(original_features_cnt, num_to_select, seed=0):
    random.seed(seed)
    return random.sample(range(0, original_features_cnt - 1), num_to_select)
