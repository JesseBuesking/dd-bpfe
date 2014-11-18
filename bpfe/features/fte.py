

from bpfe import load


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


def info():
    d = dict()
    for label, data in load.generate_training_rows():
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
