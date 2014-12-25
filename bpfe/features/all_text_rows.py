# 86.54%  346391 :
#  4.42%   17697 : all campus payroll
#  3.63%   14537 : instruction and curriculum
#  1.38%    5542 : transportation department
#  0.85%    3405 : child nutrition
#  0.54%    2173 : custodial department
#  0.19%     773 : athletic department
#  0.12%     474 : finance department
#  0.11%     438 : writing teams
#  0.10%     384 : performing arts department
# ...
#
# total entries: 400277
# unique entries: 173
from collections import Counter
from os.path import dirname
from pprint import pprint
import re
import sys
import time
from bpfe.config import Settings, ChunkSettings
from bpfe.entities import Data

import bpfe.load as load
from bpfe.text_to_vec import TextVectorizer
from bpfe.text_transform import transform
from bpfe.util import sentence_splitter


def info():
    s = Settings()
    cs = ChunkSettings(sys.maxint, sys.maxint, sys.maxint, sys.maxint)
    s.chunks = cs
    # train = reduce(
    #     lambda agg, x: agg + x, [i for i in load.gen_train(s)], [])
    # validate = reduce(
    #     lambda agg, x: agg + x, [i for i in load.gen_validate(s)], [])
    # test = reduce(
    #     lambda agg, x: agg + x, [i for i in load.gen_test(s)], [])
    submission = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_submission(s)], [])
    # _info(train + validate + test + submission)
    _info(submission)


def _info(rows):

    seen = set()
    c = Counter()
    sentences = dict()
    start = time.clock()
    replacements = 0
    fname = '{}\data\cleanup.txt'.format(
        dirname(dirname(dirname(__file__)))
    )
    with open(fname, 'w') as ifile:
        ifile.write('total rows: {}\n'.format(len(rows)))
        for idx, (data, label) in enumerate(rows):
            for attr in Data.text_attributes:
                value = getattr(data, attr)

                if value == '':
                    continue

                # for word in re.split('\s*-+\s*|\s+', value):
                #     if word not in words:
                #         words[word] = []
                #     words[word].append((attr, value))
                #     c[word] += 1
                original, value = transform(value)
                if value not in seen and original != value:
                    replacements += 1
                    seen.add(value)
                    ifile.write(
                        'idx: {} "{}" | "{}"\n'.format(idx, original, value)
                    )

                for word in sentence_splitter(value):
                    if word not in sentences:
                        sentences[word] = []

                    if len(sentences[word]) < 25:
                        if not any([i[1] == value for i in sentences[word]]):
                            sentences[word].append((attr, value))

                    if word != '':
                        c[word] += 1
        ifile.write('elapsed: {}\n'.format(time.clock() - start))
        ifile.write('replacements: {}\n'.format(replacements))

        ifile.write('total words: {}\n'.format(len(c)))
        ifile.write(str(c.most_common(10)) + '\n')

        # ordered = sorted(list(c))
        for w, count in c.most_common(sys.maxint):
            ifile.write('{} ({}):\n'.format(w, count))

            for attr, s in sentences[w]:
                ifile.write('  {}\n'.format((attr, s)))


    # tups = [(i, v) for i, v in d.iteritems()]
    # tups.sort(key=lambda x: (x[1], x[0]), reverse=True)
    #
    # tots = sum([v for _, v in tups])
    # num_unique = len(tups)
    # i = 0
    # for key, value in tups:
    #     percent = round(value/float(tots) * 100, 2)
    #     print('{:>5}% {:>7} : {}'.format(
    #         '{:0.2f}'.format(percent),
    #         value,
    #         key
    #     ))
    #     i += 1
    #     if i > 100:
    #         break
    #
    # print('')
    # print('total entries: {}'.format(tots))
    # print('unique entries: {}'.format(num_unique))
