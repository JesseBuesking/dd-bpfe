

import math
from bpfe.config import LABELS, FLAT_LABELS


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


def mean_squared(labels, predictions):
    pass
