

import pandas as pd
# import matplotlib as mpl
# mpl.use('svg')
import matplotlib.pyplot as plt


def plot_train_vs_validation(scores, amt):
    d1 = pd.DataFrame(
        [(i[1], i[2]) for i in scores['training']],
        index=[i[0] for i in scores['training']],
        columns=['correct',  'total']
    )
    d2 = pd.DataFrame(
        [(i[1], i[2]) for i in scores['validation']],
        index=[i[0] for i in scores['validation']],
        columns=['correct',  'total']
    )
    d1['percent'] = d1['correct'] / d1['total']
    d2['percent'] = d2['correct'] / d2['total']

    ax = plt.figure(figsize=(12, 6), dpi=120).add_subplot(111)
    d1.plot(kind='line', label='training', ax=ax, y='percent')
    d2.plot(kind='line', label='validation', ax=ax, y='percent')

    lines = ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc='lower right')

    ax.set_title('train vs validation for 80/20 split on {} rows'.format(amt))

    plt.show()
