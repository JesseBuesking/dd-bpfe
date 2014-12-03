# 86.57%  346531 :
#  2.69%   10762 : regular instruction
#  2.37%    9481 : basic educational services - district objective
#  1.78%    7106 : undistributed
#  0.72%    2892 : special education instruction
#  0.54%    2157 : regular salary
#  0.47%    1899 : regular instructional support
#  0.46%    1852 : transportation - second runs
#  0.35%    1383 : idea part b
#  0.31%    1257 : transportation - bus drivers
#  0.27%    1072 : office of principal
#  0.21%     827 : title i - teachers
#  0.19%     779 : career & technical instruction
#  0.16%     648 : operation of plant
#  0.15%     584 : race to the top
#  0.13%     527 : education jobs program
#  0.11%     454 : extra duty
#  0.09%     358 : teacher advancement program (tap)
# ...
#
# total entries: 400277
# unique entries: 236


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
        val = d.setdefault(data.text_4, 0)
        d[data.text_4] = val + 1

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
