

from bpfe import load


# 33.85%  135513 :
# 23.02%   92136 : professional-instructional
# 12.06%   48273 : undesignated
#  3.25%   13015 : crafts, trades, and services
#  3.16%   12639 : paraprofessional
#  2.97%   11875 : teacher bachelor
#  2.40%    9622 : substitute teacher
#  1.61%    6437 : office/administrative support
#  1.46%    5842 : professional-other
#  1.28%    5114 : teacher master
#  1.26%    5062 : bus driver
#  0.98%    3924 : teacher
#  0.87%    3471 : administrator
#  0.79%    3171 : time card certifiedaddl
#  0.72%    2868 : degreed substitute
# ...
#
# total entries: 400277
# unique entries: 489


def info():
    d = dict()
    for label, data in load.generate_training_rows():
        val = d.setdefault(data.position_extra, 0)
        d[data.position_extra] = val + 1

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
