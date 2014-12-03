# 77.96%  312060 :
#  4.15%   16599 : teacher subs
#  1.47%    5871 : food services
#  1.06%    4251 : general education
#  0.99%    3945 : transportation
#  0.96%    3860 : custodial-schools
#  0.72%    2878 : teacher learning & leadership
#  0.68%    2710 : teacher
#  0.65%    2610 : severe disabilities
#  0.62%    2483 : maintenance
#  0.57%    2264 : afterschool programs
#  0.54%    2170 : adult voc ed opportunity
#  0.48%    1919 : math/science
# ...
#
# total entries: 400277
# unique entries: 287


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
        val = d.setdefault(data.text_2, 0)
        d[data.text_2] = val + 1

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
