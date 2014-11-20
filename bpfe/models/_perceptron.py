"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""


from collections import defaultdict
from bpfe.config import LABEL_MAPPING


class AveragedPerceptron(object):
    """ An averaged perceptron. """

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = dict()
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
        scores = defaultdict(float)
        for feat, value in features.items():
            if value == 0:
                continue

            weights = self.weights.get(feat)
            if weights is None:
                continue

            for label, weight in weights.items():
                scores[label] += value * weight

        # Do a secondary alphabetic sort, for stability
        ret = dict()
        for key in LABEL_MAPPING.keys():
            ret[key] = max(
                self.classes[key],
                key=lambda x: (scores[x], x)
            )
        return ret

    def update(self, truth, guess, features):
        """ Update the feature weights. """
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        for real_key, key in LABEL_MAPPING.iteritems():
            t = getattr(truth, key)
            g = guess.get(real_key)
            if t == g:
                continue
            for f in features.keys():
                weights = self.weights.setdefault(f, {})
                upd_feat(t, f, weights.get(t, 0.0), 1.0)
                upd_feat(g, f, weights.get(g, 0.0), -1.0)
        return None

    def average_weights(self):
        """ Average weights from all iterations. """
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for klass, weight in weights.items():
                param = (feat, klass)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[klass] = averaged
            self.weights[feat] = new_feat_weights
        return None
