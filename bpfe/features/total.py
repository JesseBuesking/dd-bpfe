# 1.14%    4555 :
#  0.01%      57 : 0.22
#  0.01%      55 : 0.12
#  0.01%      54 : 0.21
#  0.01%      54 : 0.06
#  0.01%      52 : -0.1
#  0.01%      51 : 109.99793846799999
#  0.01%      51 : 0.28
#  0.01%      50 : 0.27
#  0.01%      50 : 0.24
#  0.01%      49 : -0.14
# ...
#
# total entries: 400277
# unique entries: 286501


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
        if isfloat(data.total):
            tot = float(data.total)
            if -10000 <= tot <= 120000:
            # tot = min(tot, 120000)
            # tot = max(tot, -10000)
                values.append(tot)
        val = d.setdefault(data.total, 0)
        d[data.total] = val + 1

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
