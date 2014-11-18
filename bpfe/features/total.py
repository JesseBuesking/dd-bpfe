

from bpfe import load


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


def info():
    d = dict()
    for label, data in load.generate_training_rows():
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
