

import math
from bpfe.config import LABELS, FLAT_LABELS
import numpy as np


def multi_multi(actuals, predictions):
    k_tot = 0
    d = dict()
    for f_idx, flat in enumerate(FLAT_LABELS):
        d[flat] = f_idx

    for K, Cs in LABELS.iteritems():
        n_len = len(actuals)
        n_tot = 0
        for n in range(n_len):
            c_tot = 0
            for c in Cs:
                tup = (K, c)
                idx = d[tup]
                c_tot += _ll(actuals[n][idx], predictions[n][idx])
            n_tot += c_tot
        n_tot /= float(n_len)
        k_tot += -n_tot

    return k_tot / float(len(LABELS))


def _ll(actual, predicted):
    epsilon = 1e-15
    predicted = max(epsilon, predicted)
    predicted = min(1-epsilon, predicted)
    return actual * math.log(predicted)


def multi_multi_correct(actuals, predictions):
    k_tot = 0
    d = dict()
    for f_idx, flat in enumerate(FLAT_LABELS):
        d[flat] = f_idx

    for K, Cs in LABELS.iteritems():
        n_len = len(actuals)
        n_tot = 0
        for n in range(n_len):
            c_tot = 0
            for c in Cs:
                tup = (K, c)
                idx = d[tup]
                c_tot += _ll_correct(actuals[n][idx], predictions[n][idx])
            n_tot += c_tot
        n_tot /= float(n_len)
        k_tot += -n_tot

    return k_tot / float(len(LABELS))


def _ll_correct(actual, predicted):
    epsilon = 1e-15
    predicted = max(epsilon, predicted)
    predicted = min(1-epsilon, predicted)
    return actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted)


BOX_PLOTS_COLUMN_INDICES = [range(37),
                            range(37,48),
                            range(48,51),
                            range(51,76),
                            range(76,79),
                            range(79,82),
                            range(82,87),
                            range(87,96),
                            range(96,104)]


def multi_multi_log_loss(predicted, actual, class_column_indices, eps=1e-15):
    """
    Multi class, multi-label version of Logarithmic Loss metric.

    :param predicted: a 2d numpy array of the predictions that are probabilities
     [0, 1]
    :param actual: a 2d numpy array of the same shape as your predictions. 1 for
     the actual labels, 0 elsewhere
    :return: The multi-multi log loss score for this set of predictions
    """
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = predicted[:, this_class_indices]

        # normalize so probabilities sum to one (unless sum is zero, then we
        # clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)

        actual_k = actual[:, this_class_indices]

        # shrink predictions
        y_hats = np.clip(preds_k, eps, 1 - eps)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(class_scores)
