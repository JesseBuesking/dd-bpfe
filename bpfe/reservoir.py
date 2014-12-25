

import random
from unidecode import unidecode


def reservoir(data, seed, amt):
    random.seed(seed)
    sample = []
    for i, line in enumerate(data):
        new_line = []
        # is_unicode = False
        for col in line:
            if col != unidecode(col):
                u = unicode(col, 'utf-8')
                u = u.replace(u'\xc3\u0192\xc2\u0192', '')
                val = u.encode('raw_unicode_escape')
                val = val.replace('\\u2013', '--')
                uval = unidecode(val)
                print(col, unicode(col, 'utf-8'), unidecode(col), val)
                print('====  ' + uval)
                col = uval
                # is_unicode = True
            new_line.append(col)
        # if is_unicode:
        #     print('line: {}'.format(line))
        #     print('nlin: {}'.format(new_line))
        line = new_line

        if i < amt:
            sample.append(line)
        elif i >= amt and random.random() < amt / float(i+1):
            replace = random.randint(0, len(sample)-1)
            sample[replace] = line

    return sample
