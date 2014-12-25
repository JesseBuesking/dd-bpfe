

import collections
import re
import sys
from bpfe import load
from bpfe.config import Settings, ChunkSettings
import pandas as pd
import matplotlib.pyplot as plt
from bpfe.entities import Data


class SpellCorrector(object):

    def __init__(self, word_gen, min_count=None, min_len=None):
        self.NWORDS = self._train(word_gen)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

        if min_count is not None:
            assert isinstance(min_count, int)

            for word in self.NWORDS.keys():
                if self.NWORDS[word] < min_count:
                    del self.NWORDS[word]

        if min_len is not None:
            assert isinstance(min_len, int)

            for word in self.NWORDS.keys():
                if len(word) < min_len:
                    del self.NWORDS[word]

    def _train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def _edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _known_edits2(self, eword):
        return set(
            e2 for e1 in eword for e2 in self._edits1(e1) if e2 in self.NWORDS
        )

    def _known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        candidates = self._known([word])
        if len(candidates) <= 0:
            eword = self._edits1(word)
            candidates = self._known(eword)

        if len(candidates) <= 0:
            # noinspection PyUnboundLocalVariable
            candidates = self._known_edits2(eword)

        if len(candidates) <= 0:
            candidates = [word]

        return max(candidates, key=self.NWORDS.get)


if __name__ == '__main__':
    s = Settings()
    cs = ChunkSettings(sys.maxint, sys.maxint, sys.maxint, sys.maxint)
    # cs = ChunkSettings(1, 1, 1, 1)
    s.chunks = cs
    train = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_train(s)], [])
    validate = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_validate(s)], [])
    test = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_test(s)], [])
    submission = reduce(
        lambda agg, x: agg + x, [i for i in load.gen_submission(s)], [])
    _all = train + validate + test + submission

    def word_gen(data):
        for idx, (data, label) in enumerate(data):
            for attr in Data.text_attributes:
                value = getattr(data, attr)

                if value == '':
                    continue

                for word in re.findall('[a-zA-Z]+', value):
                # for word in re.split('\s+', value):
                    yield word.lower()

    print('creating SpellCorrector')
    sc = SpellCorrector(word_gen(_all), min_count=3, min_len=6)
    print('len: {}'.format(len(sc.NWORDS)))
    print('len: {}'.format(sorted([i for i in sc.NWORDS])))

    print('running corrections')
    total = 0
    for word in word_gen(_all):
        cword = sc.correct(word)
        if cword != word:
            total += 1
            if total < 10:
                print(word, cword)
    print('total: {}'.format(total))

    #                    c
    # count    5651.000000
    # mean     1522.540789
    # std     12129.876316
    # min         2.000000
    # 25%         4.000000
    # 50%        17.000000
    # 75%       139.500000
    # max    393325.000000

    # df = pd.DataFrame([(w, c) for w, c in sc.NWORDS.items()], columns=['w', 'c'])
    # print(df.describe())
    # df.hist()
    # plt.show()
