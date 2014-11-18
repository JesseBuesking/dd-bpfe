

from bpfe import load


# 59.51%  238223 :
# 16.37%   65524 : school
#  1.94%    7749 : admin. services
#  1.50%    6005 : unallocated
#  1.37%    5485 : special education
#  1.33%    5309 : undistributed
#  1.22%    4867 : transportation
#  0.91%    3626 : district wide resources
#  0.76%    3044 : opportunity school
#  0.62%    2482 : teacher learning & leadership
#  0.58%    2339 : summer school
#  0.57%    2279 : garage
#  0.53%    2126 : charter
#  0.53%    2123 : math / science
# ...
#
# total entries: 400277
# unique entries: 348


def info():
    d = dict()
    for label, data in load.generate_training_rows():
        val = d.setdefault(data.location_description, 0)
        d[data.location_description] = val + 1

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
