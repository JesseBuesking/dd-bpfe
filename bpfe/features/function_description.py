# 19.21%   76890 : non-project
# 14.51%   58082 :
#  9.99%   39999 : instruction
#  4.15%   16616 : unalloc budgets/schools
#  3.33%   13317 : basic (fefp k-12)
#  3.27%   13073 : employee retirement
#  1.95%    7814 : disadvantaged youth *
#  1.58%    6311 : ela e-teaching sheltered eng
#  1.51%    6043 : instruction and curriculum development services *
#  1.24%    4957 : inst staff training svcs
#  1.08%    4325 : transportation
#  1.00%    4009 : title i a - arra
# ...
#
# total entries: 400277
# unique entries: 648


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
        val = d.setdefault(data.function_description, 0)
        d[data.function_description] = val + 1

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
