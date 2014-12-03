# 27.00%  108070 :
# 16.21%   64896 : regular instruction
#  8.16%   32669 : employee benefits
#  7.64%   30592 : instructional staff
#  4.65%   18609 : regular pay
#  2.44%    9778 : special education
#  2.29%    9161 : operation and maint of plant
#  1.46%    5857 : food services operations
#  1.39%    5575 : school administration
#  1.22%    4864 : students
#  1.21%    4858 : central
#  1.17%    4674 : miscellaneous
#  1.07%    4277 : school based management
#  1.01%    4040 : student transportation
#  0.86%    3452 : extended days
#  0.74%    2955 : title i
#  0.72%    2881 : overtime
#  0.68%    2713 : addl regular pay-not smoothed
#  0.62%    2480 : esea title i
#  0.61%    2450 : teacher lead
#  0.52%    2075 : school recognition
# ...
#
# total entries: 400277
# unique entries: 1387


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
        val = d.setdefault(data.text_1, 0)
        d[data.text_1] = val + 1

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
