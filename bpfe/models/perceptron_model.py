# -*- coding: utf-8 -*-


from __future__ import absolute_import
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'US')
import random
from collections import defaultdict
import pickle
import time
from bpfe import scoring
from bpfe.config import LABELS, LABEL_MAPPING, FLAT_LABELS, INPUT_CODES
from bpfe.models._perceptron import AveragedPerceptron


class PerceptronModel(object):
    """ Greedy Averaged Perceptron predictor. """

    __slots__ = ('model', 'classes', 'scores', 'seed', 'iterations', 'start',
                 'end', 'amt')

    def __init__(self):
        self.model = AveragedPerceptron()
        self.classes = dict()
        self.scores = []

    def predict(self, data):
        """ Makes a prediction. """

        features = self._get_features(data)
        return self.model.predict(features)

    def predict_proba(self, data):
        """ Makes a prediction. """

        features = self._get_features(data)
        return self.model.predict_proba(features)

    def train(self, training_data, test_data, amt, save_loc=None, nr_iter=5,
              seed=1):
        """
        Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param training_data: the data to be trained on
        :param test_data: data to validate on
        :param save_loc: if not ``None``, saves a pickled model in this location
        :param nr_iter: number of training iterations
        """
        self.amt = amt
        self.seed = seed
        self.iterations = nr_iter
        self.start = datetime.utcnow()
        random.seed(seed)
        for key, values in LABELS.iteritems():
            self.classes[key] = set()
            for value in values:
                self.classes[key].add(value)

        self.print_header()

        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            start = time.clock()
            c_train = 0
            n_train = 0
            for label, data in training_data:
                feats = self._get_features(data)
                guess = self.model.predict(feats)
                self.model.update(label, guess, feats)
                for real_key, key in LABEL_MAPPING.iteritems():
                    c_train += guess.get(real_key) == getattr(label, key)
                    n_train += 1
            random.shuffle(training_data)

            c_test = 0
            n_test = 0
            actuals = []
            predictions = []
            for label, data in test_data:
                feats = self._get_features(data)
                guess = self.model.predict(feats)
                for real_key, key in LABEL_MAPPING.iteritems():
                    c_test += guess.get(real_key) == getattr(label, key)
                    n_test += 1

                actuals.append(
                    PerceptronModel.label_output(label)[1:]
                )
                predictions.append(
                    PerceptronModel.prediction_output(data, guess)[1:]
                )

            ll = scoring.multi_multi(actuals, predictions)
            ll_correct = scoring.multi_multi_correct(actuals, predictions)
            end = time.clock()

            score = (
                iter_, c_train, n_train, c_test, n_test, ll, ll_correct,
                _td(end - start)
            )
            self.print_score(score)
            self.scores.append(score)

        self.model.average_weights()
        self.end = datetime.utcnow()
        self.save(save_loc)
        return None

    def print_scores(self):
        self.print_header()
        for score in self.scores:
            self.print_score(score)

    def print_header(self):
        print('')
        header = ''.join([
            ' ' * 23,
            'train score',
            ' ' * 21,
            'test score',
            ' ' * 5,
            'mc log loss',
            ' ' * 5,
            'mc log loss*',
            ' ' * 5,
            'elapsed time'
        ])
        print(header)
        print('-' * len(header))

    def print_score(self, score):
        print('{:02d}: {:>20} = {}% {:>20} = {}% {:>15} {:>16} {:>16}'.format(
            score[0],
            '{}/{}'.format(
                locale.format('%d', score[1], grouping=True),
                locale.format('%d', score[2], grouping=True)
            ),
            '{:.03f}'.format(_pc(score[1], score[2])),
            '{}/{}'.format(
                locale.format('%d', score[3], grouping=True),
                locale.format('%d', score[4], grouping=True)
            ),
            '{:.03f}'.format(_pc(score[3], score[4])),
            '{:.03f}'.format(score[5]),
            '{:.03f}'.format(score[6]),
            '{}'.format(score[7])
        ))

    def save(self, loc):
        """ Save the pickled model weights. """
        if loc is None:
            return None

        return pickle.dump((
            self.model.weights,
            self.classes,
            self.scores,
            self.seed,
            self.iterations,
            self.start,
            self.end,
            self.amt
        ), open(loc, 'wb'), -1)

    def load(self, loc):
        """ Load the pickled model weights. """
        try:
            pkl = pickle.load(open(loc, 'rb'))
        except IOError:
            raise Exception("missing {} file".format(loc))
        self.model.weights, self.classes, self.scores, self.seed, \
            self.iterations, self.start, self.end, self.amt = pkl
        self.model.classes = self.classes
        return None

    def _get_features(self, data):
        """
        Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        """
        def add(name, *args):
            if len(args) > 0:
                if isinstance(args[0], list):
                    for l in args[0]:
                        features[' '.join([name, l])] += 1
                else:
                    features[' '.join(((name,) + tuple(args)))] += 1
            else:
                features[name] += 1

        features = defaultdict(int)
        for key in data.attributes:
            name = INPUT_CODES[key]
            add(name, getattr(data, key))

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

    def __str__(self):
        ret = ''
        ret += 'iterations: {}\n'.format(self.iterations)
        ret += 'amt: {}\n'.format(self.amt)
        ret += 'seed: {}\n'.format(self.seed)
        ret += 'start: {}\n'.format(self.start)
        ret += 'end: {}'.format(self.end)
        return ret


def _pc(n, d):
    return round((float(n) / d) * 100, 2)


def _td(value):
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)

    return '%02d:%02d:%02d' % (hours, minutes, seconds)
