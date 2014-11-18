

from bpfe import load


# 77.12%  308674 :
#  2.29%    9159 : extra duty pay/overtime for support personnel
#  2.18%    8724 : certificated employees salaries and wages
#  1.82%    7285 : salaries and wages for teachers and other professi
#  1.58%    6327 : salaries and wages for substitute teachers
#  1.46%    5833 : general supplies *
#  1.21%    4824 : salaries or wages for support personnel
#  0.82%    3273 : supplies and materials
#  0.76%    3045 : extended day
#  0.68%    2730 : extra duty wages
# ...
#
# total entries: 400277
# unique entries: 167


def info():
    d = dict()
    for label, data in load.generate_training_rows():
        val = d.setdefault(data.sub_object_description, 0)
        d[data.sub_object_description] = val + 1

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
