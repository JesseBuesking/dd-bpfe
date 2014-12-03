# 55.04%  220313 :
# 23.60%   94462 : regular
# 17.69%   70812 : n/a
#  1.94%    7755 : turnaround
#  1.32%    5299 : alternative
#  0.17%     664 : charter
#  0.16%     623 : new/closed schl
#  0.01%      60 : other materials and supplies
#  0.01%      42 : supplies for instruction
#  0.01%      41 : employee travel
#  0.01%      39 : staff development
# ...
#
# total entries: 400277
# unique entries: 36


import bpfe.load as load


def info(num_chunks=None):
    train = [i for i in load.gen_train(num_chunks)]
    validate = [i for i in load.gen_validate(num_chunks)]
    test = [i for i in load.gen_test(num_chunks)]
    submission = [i for i in load.gen_submission(num_chunks)]
    _info(train + validate + test + submission)


def _info(rows):
    d = dict()
    for data, label in rows:
        val = d.setdefault(data.text_3, 0)
        d[data.text_3] = val + 1

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
