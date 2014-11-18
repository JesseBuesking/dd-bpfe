

from bpfe import load


# 30.81%  123327 : general fund
# 23.34%   93422 :
#  6.72%   26895 : operations
#  3.39%   13562 : federal gdpg fund - fy
#  2.56%   10238 : support services - instructional staff
#  2.52%   10106 : special instruction
#  2.49%    9975 : district special revenue funds
#  2.35%    9393 : local
#  1.86%    7458 : arra - stimulus
#  1.85%    7386 : community services
#  1.49%    5964 : mill levy
#  1.45%    5810 : food service fund
# ...
#
# total entries: 400277
# unique entries: 268


def info():
    d = dict()
    for label, data in load.generate_training_rows():
        val = d.setdefault(data.subfund_description, 0)
        d[data.subfund_description] = val + 1

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
