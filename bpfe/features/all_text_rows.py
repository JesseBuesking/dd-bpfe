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
from pprint import pprint
import sys
from bpfe.config import Settings, ChunkSettings

import bpfe.load as load
from bpfe.text_to_vec import TextVectorizer


def info():
    s = Settings()
    cs = ChunkSettings(sys.maxint, sys.maxint, sys.maxint, sys.maxint)
    s.chunks = cs
    train = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_train(s)], [])
    validate = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_validate(s)], [])
    test = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_test(s)], [])
    submission = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_submission(s)], [])
    _info(train + validate + test + submission)


def _info(rows):
    d = dict()
    text_attributes = [
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'function_description', 'position_extra',
        'text_4', 'text_2', 'text_3', 'fund_description', 'text_1'
    ]

    example_info = dict()

    for data, label in rows:
        for attr in text_attributes:
            value = getattr(data, attr)
            val = d.setdefault(value, 0)
            d[value] = val + 1

            if value not in example_info:
                example_info[value] = []
            example_info[value].append((attr, value))

    tv = TextVectorizer()
    tv.fit(d.keys())
    top_n = sorted(
        [(word, count) for word, count in tv.value_indices.items()],
        key=lambda (w, c): c,
        reverse=True
    )

    print('total words: {}'.format(len(top_n)))
    print(top_n[:10])

    ordered = sorted([i[0] for i in top_n])
    for word in ordered:
        print('{}:'.format(word))

        idx = 0
        for sentence, infos in example_info.items():
            if ' ' + word + ' ' in sentence or \
               (' ' + word in sentence and sentence.endswith(word)) or \
               (word + ' ' in sentence and sentence.startswith(word)):
                print('  {}'.format(infos[:2]))
                idx += 1
                if idx >= 3:
                    break

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
