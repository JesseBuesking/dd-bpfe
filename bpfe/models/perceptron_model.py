# -*- coding: utf-8 -*-


from __future__ import absolute_import
import random
from collections import defaultdict
import pickle
import time
from bpfe import scoring
from bpfe.config import LABELS, LABEL_MAPPING, FLAT_LABELS
from bpfe.models._perceptron import AveragedPerceptron


class PerceptronModel(object):
    """ Greedy Averaged Perceptron predictor. """

    def __init__(self):
        self.model = AveragedPerceptron()
        self.classes = dict()
        self.scores = {
            'training': [],
            'validation': []
        }

    def predict(self, data):
        """ Makes a prediction. """

        features = self._get_features(data)
        return self.model.predict(features)

    def train(self, training_data, test_data, save_loc=None, nr_iter=5, seed=1):
        """
        Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param training_data: the data to be trained on
        :param test_data: data to validate on
        :param save_loc: if not ``None``, saves a pickled model in this location
        :param nr_iter: number of training iterations
        """
        random.seed(seed)
        for key, values in LABELS.iteritems():
            self.classes[key] = set()
            for value in values:
                self.classes[key].add(value)

        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            start = time.clock()
            c = 0
            n = 0
            for label, data in training_data:
                feats = self._get_features(data)
                guess = self.model.predict(feats)
                self.model.update(label, guess, feats)
                for real_key, key in LABEL_MAPPING.iteritems():
                    c += guess.get(real_key) == getattr(label, key)
                    n += 1
            random.shuffle(training_data)
            print("iter {}: {}/{}={}%".format(iter_, c, n, _pc(c, n)))
            self.scores['training'].append((iter_, c, n))

            c = 0
            n = 0
            actuals = []
            predictions = []
            for label, data in test_data:
                feats = self._get_features(data)
                guess = self.model.predict(feats)
                for real_key, key in LABEL_MAPPING.iteritems():
                    c += guess.get(real_key) == getattr(label, key)
                    n += 1

                actuals.append(
                    PerceptronModel.label_output(label)[1:]
                )
                predictions.append(
                    PerceptronModel.prediction_output(data, guess)[1:]
                )

            ll = scoring.multi_multi(actuals, predictions)

            print("  validation: {}/{}={}% (ll: {})".format(
                c,
                n,
                _pc(c, n),
                round(ll, 4)
            ))
            end = time.clock()
            print("  elapsed: {}".format(end-start))
            self.scores['validation'].append((iter_, c, n, ll))

        self.model.average_weights()
        self.save(save_loc)
        return None

    def save(self, loc):
        """ Save the pickled model weights. """
        if loc is None:
            return None

        return pickle.dump((
            self.model.weights,
            self.classes,
            self.scores
        ), open(loc, 'wb'), -1)

    def load(self, loc):
        """ Load the pickled model weights. """
        try:
            w_c_s = pickle.load(open(loc, 'rb'))
        except IOError:
            raise Exception("missing {} file".format(loc))
        self.weights, self.classes, self.scores = w_c_s
        self.model.classes = self.classes
        return None

    def _get_features(self, data):
        """
        Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        """
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        # i += len(self.START)
        features = defaultdict(int)
        for key in data.__slots__:
            add(key, getattr(data, key))
        # it's useful to have a constant feature which acts sort of like a prior
        add('bias')
        return features

    @staticmethod
    def label_output(labels):
        pred_tups = set()
        for raw_key, key in LABEL_MAPPING.iteritems():
            pred_tups.add((raw_key, getattr(labels, key)))
        r = [0.0]
        for t in FLAT_LABELS:
            if t in pred_tups:
                r.append(1.0)
            else:
                r.append(0.0)
        return r

    @staticmethod
    def prediction_output(data, prediction):
        pred_tups = set()
        for raw_key in LABEL_MAPPING.keys():
            pred_tups.add((raw_key, prediction[raw_key]))
        r = [data.id]
        for t in FLAT_LABELS:
            if t in pred_tups:
                r.append(1.0)
            else:
                r.append(0.0)
        return r


def _pc(n, d):
    return round((float(n) / d) * 100, 2)
