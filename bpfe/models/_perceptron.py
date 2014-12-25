"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""


from collections import defaultdict
import numpy as np
from sklearn.preprocessing import MinMaxScaler


eps = 1e-15
mms = MinMaxScaler((eps, 1-eps))


class AveragedPerceptron(object):
    """ An averaged perceptron. """

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = list()
        self.learning_rate = 0.1
        # The accumulated values, for the averaging. These will be keyed by
        # feature/class tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/class tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        """
        Dot-product the features and current weights and return the best label.
        """
        scores = self.predict_proba(features)
        return self.classes[np.argmax(scores)]

    def predict_proba(self, features):
        """
        Dot-product the features and current weights and return the best label.
        """
        scores = np.zeros(len(self.classes))
        for feat, value in features.items():
            if value == 0:
                continue

            weights = self.weights.get(feat)
            if weights is None:
                continue

            scores += (value * weights)

        # scores = (scores - scores.mean()) / scores.std()
        # scores[np.isnan(scores)] = eps
        scores = mms.fit_transform(scores)

        return scores

    def update(self, truth, guess, features):
        def upd_feat(f, w, v):
            self._totals[f] += (self.i - self._tstamps[f]) * w
            self._tstamps[f] = self.i
            self.weights[f] = w + v

        diff = np.array(truth) - np.array(guess)
        diff *= self.learning_rate

        self.i += 1
        for f in features.keys():
            weights = self.weights.setdefault(f, np.zeros(len(self.classes)))
            upd_feat(f, weights, diff)

    def average_weights(self):
        """ Average weights from all iterations. """
        for feat, weight in self.weights.items():
            total = self._totals[feat]
            total += (self.i - self._tstamps[feat]) * weight
            averaged = total / float(self.i)
            self.weights[feat] = averaged
        return None
