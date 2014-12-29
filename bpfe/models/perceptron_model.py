# -*- coding: utf-8 -*-


from __future__ import absolute_import
from datetime import datetime
import locale
import re
from sklearn.metrics import log_loss
import sys
from bpfe.entities import Data
from bpfe.feature_engineering import all_ngrams


locale.setlocale(locale.LC_ALL, 'US')
import random
from collections import defaultdict
import pickle
import time
import numpy as np
from bpfe import scoring
from bpfe.config import LABEL_MAPPING, FLAT_LABELS
from bpfe.models._perceptron import AveragedPerceptron
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


stop = stopwords.words('english')
sbs = SnowballStemmer('english')


def bow(string):
    for word in re.findall(
        r'GRADE=k\|k|GRADE=k\|\d+|GRADE=\d+\|\d+|\w+|\d+|\.',
        string
    ):
        yield word


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
        self.classes = set()
        random.seed(seed)
        for _, label in training_data:
            # noinspection PyUnresolvedReferences
            self.classes.add(label)
        self.classes = list(self.classes)

        self.print_header()

        self.model.classes = self.classes

        mmll_prev = float(sys.maxint)
        iter_since_improv = 0

        for iter_ in range(nr_iter):
            start = time.clock()
            c_train = 0
            for data, label in training_data:
                tmp = np.zeros(len(self.classes))
                tmp[self.classes.index(label)] = 1

                feats = self._get_features(data)
                guess = self.model.predict_proba(feats)

                c_train += np.argmax(guess) == np.argmax(tmp)

                self.model.update(tmp, guess, feats)

            preds, actuals = [], []
            for data, label in training_data:
                tmp = np.zeros(len(self.classes))
                tmp[self.classes.index(label)] = 1

                feats = self._get_features(data)
                guess = self.model.predict_proba(feats)

                preds.append(guess)
                actuals.append(tmp)

            mmll_train = scoring.multi_multi_log_loss(
                np.array(preds),
                np.array(actuals),
                np.array([range(actuals[0].shape[0])])
            )

            def improvement(x, y):
                return (x - y) / (2 * max(x, y))

            imp = improvement(mmll_prev, mmll_train)
            if imp < .005:
                print('halving the learning rate: {} -> {}'.format(
                    self.model.learning_rate,
                    self.model.learning_rate / 2
                ))
                self.model.learning_rate /= 2
                iter_since_improv += 1
            else:
                iter_since_improv = 0

            if iter_since_improv >= 10:
                print('no improvement in 10 iterations, stopping')
                break

            mmll_prev = mmll_train

            random.shuffle(training_data)

            c_test = 0
            preds, actuals = [], []
            for data, label in test_data:
                tmp = np.zeros(len(self.classes))
                tmp[self.classes.index(label)] = 1

                feats = self._get_features(data)
                guess = self.model.predict_proba(feats)

                c_test += np.argmax(guess) == np.argmax(tmp)

                preds.append(guess)
                actuals.append(tmp)

            mmll_test = scoring.multi_multi_log_loss(
                np.array(preds),
                np.array(actuals),
                np.array([range(actuals[0].shape[0])])
            )

            end = time.clock()

            score = (
                iter_, c_train, len(training_data), c_test, len(test_data),
                mmll_train, mmll_test, _td(end - start)
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
            'train log loss',
            ' ' * 2,
            'test log loss',
            ' ' * 4,
            'elapsed time'
        ])
        print(header)
        print('-' * len(header))

    def print_score(self, score):
        print('{:02d}: {:>20} = {}% {:>20} = {}% {:>18} {:>14} {:>15}'.format(
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
        def add(name, amount, *args):
            if len(args) > 0:
                if isinstance(args[0], list):
                    for l in args[0]:
                        features[' '.join([name, l])] += amount
                else:
                    features[' '.join(((name,) + tuple(args,)))] += amount
            else:
                features[name] += amount

        features = defaultdict(int)

        for attr in Data.text_attributes:
            value = data.cleaned[attr + '-mapped']
            # value = getattr(data, attr)
            b_o_w = []
            for i in bow(value):
                i = sbs.stem(i)
                b_o_w.append(i)

            ng = all_ngrams(b_o_w, 3)
            ng = [i for i in ng if i not in stop]
            add(attr, 1, ng)
            add('ng', 1, ng)

        for idx, grade in enumerate(data.grades):
            add('gr {}'.format(idx), 1, str(int(grade)))

        for idx, title in enumerate(data.title):
            add('ttl {}'.format(idx), 1, str(int(title)))

        # it's useful to have a constant feature which acts sort of like a prior
        add('bias', 1)
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
