# 86.54%  346391 :
#  4.42%   17697 : all campus payroll
#  3.63%   14537 : instruction and curriculum
#  1.38%    5542 : transportation department
#  0.85%    3405 : child nutrition
#  0.54%    2173 : custodial department
#  0.19%     773 : athletic department
#  0.12%     474 : finance department
#  0.11%     438 : writing teams
#  0.10%     384 : performing arts department
# ...
#
# total entries: 400277
# unique entries: 173


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
        val = d.setdefault(data.facility_or_department, 0)
        d[data.facility_or_department] = val + 1

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
