class SpamHeuristicModel(object):
    """A heuristic spam filter"""

    def __init__(self):
        pass

    def fit(self, x, y):
        self.weights = [0.75] * len(x[0])
        pass

    def predict(self, x):
        predictions = []

        for example in x:
            scores = [ example[i] * self.weights[i] for i in range(len(example)) ]

            if sum(scores) > 1.5:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions