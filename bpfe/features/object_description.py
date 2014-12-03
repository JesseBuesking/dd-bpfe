# 11.87%   47495 : employee benefits
#  7.93%   31761 : salaries of part time employee
#  6.19%   24784 :
#  6.08%   24319 : salaries of regular employees
#  4.84%   19381 : contra benefits
#  4.65%   18632 : salaries and wages for teachers and other professi
#  4.21%   16841 : additional/extra duty pay/stip
#  4.16%   16656 : supplies
#  3.27%   13073 : retirement contrib.
#  2.32%    9270 : regular *
# ...
#
# total entries: 400277
# unique entries: 574


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
        val = d.setdefault(data.object_description, 0)
        d[data.object_description] = val + 1

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
