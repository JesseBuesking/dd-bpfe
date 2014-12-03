# 26.87%  107541 :
#  7.73%   30957 : teacher, elementary
#  5.86%   23450 : teacher, short term sub
#  3.81%   15235 : (blank)
#  2.25%    8994 : teacher, secondary (high)
#  2.18%    8746 : teacher,retrd shrt term sub
#  2.13%    8517 : teacher, regular
#  1.78%    7115 : teacher substitute pool
#  1.75%    7018 : sub teacher all
#  1.73%    6912 : teacher secondary (middle)
#  1.31%    5224 : teacher
#  0.93%    3704 : teacher, super sub d/d
#  0.83%    3342 : transportation,bus drivers,reg
#  0.68%    2729 : food service worker ii
#  0.59%    2368 : teacher,spec ed center prg
#  0.58%    2315 : bus driver
#  0.57%    2297 : early childhood education
#  0.48%    1917 : cafeteria employee, food serv.
#  0.48%    1913 : elementary spec ed para
#  0.48%    1909 : teacher, intervention
#  0.47%    1895 : teacher-elementary
#  0.47%    1887 : custodial helper - day
#  0.47%    1884 : custodian
# ...
#
# total entries: 400277
# unique entries: 3286


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
        val = d.setdefault(data.job_title_description, 0)
        d[data.job_title_description] = val + 1

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
