# 68.50%  274206 :
#  8.94%   35788 : 1.0
#  7.83%   31338 : 0.0
#  2.03%    8130 : 0.004310344827590001
#  0.68%    2735 : 0.00215517241379
#  0.57%    2293 : 0.00862068965517
#  0.50%    2020 : 0.025
#  0.46%    1825 : 0.5
#  0.24%     941 : 0.0129310344828
#  0.18%     712 : 0.006465517241380001
#  0.12%     479 : 0.0172413793103
#  0.10%     398 : 0.25
#  0.10%     383 : 0.75
# ...
#
# total entries: 400277
# unique entries: 21004


import bpfe.load as load
from bpfe.util import isfloat
import matplotlib.pyplot as plt
import seaborn as sns


def info(num_chunks=None):
    train = [i for i in load.gen_train(num_chunks)]
    validate = [i for i in load.gen_validate(num_chunks)]
    test = [i for i in load.gen_test(num_chunks)]
    submission = [i for i in load.gen_submission(num_chunks)]
    _info(train + validate + test + submission)


def _info(rows):
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (10, 5)})

    values = []
    d = dict()
    for data, label in rows:
        if isfloat(data.fte):
            fte = float(data.fte)
            if 0.0 <= fte <= 1.0:
            # fte = min(fte, 1.0)
            # fte = max(fte, 0.0)
                values.append(fte)
        val = d.setdefault(data.fte, 0)
        d[data.fte] = val + 1

    tups = [(i, v) for i, v in d.iteritems()]
    tups.sort(key=lambda x: (x[1], x[0]), reverse=True)

    tots = sum([v for _, v in tups])
    num_unique = len(tups)
    i = 0
    for key, value in tups:
        percent = round(value/float(tots) * 100, 2)
        print('{:>5}% {:>7} : {}'.format(
            '{:0.2f}'.format(percent),
            value,
            key
        ))
        i += 1
        if i > 100:
            break

    print('')
    print('total entries: {}'.format(tots))
    print('unique entries: {}'.format(num_unique))

    sns.distplot(
        values,
        25,
        kde_kws={
            "color": "seagreen",
            "lw": 3,
            "label": "KDE"
        },
        hist_kws={
            "histtype": "stepfilled",
            "color": "slategray"
        }
    )
    plt.show()
