# 49.32%  197400 :
# 14.29%   57212 : general fund
#  8.36%   33467 : general operating fund
#  5.08%   20333 : general purpose school
#  4.99%   19967 : title i - disadvantaged children/targeted assistance
#  1.97%    7877 : general
#  1.72%    6885 : "title part a improving basic programs"
#  1.58%    6335 : special trust
#  1.36%    5424 : school federal projects
#  0.89%    3544 : fed thru state-cash advance
#  0.85%    3405 : national school breakfast and lunch program
#  0.65%    2616 : food services
# ...
#
# total entries: 400277
# unique entries: 138


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
        val = d.setdefault(data.fund_description, 0)
        d[data.fund_description] = val + 1

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
