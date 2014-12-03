# 23.89%   95617 :
#  8.20%   32829 : general elementary education
#  8.16%   32669 : employee benefits
#  5.38%   21521 : instructional staff training
#  4.63%   18547 : undistributed
#  3.45%   13825 : instruction - regular
#  3.28%   13143 : misc
#  2.65%   10625 : general high school education
#  2.65%   10593 : basic educational services
#  2.13%    8520 : "title i, part a schoolwide activities related to state comp
# ...
#
# total entries: 400277
# unique entries: 417


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
        val = d.setdefault(data.program_description, 0)
        d[data.program_description] = val + 1

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
