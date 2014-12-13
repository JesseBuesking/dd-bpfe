

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np


remove_punc_keep_hyph_tokenizer = RegexpTokenizer(r'\w+(-\w+)*')
sw = stopwords.words('english')


class BinaryVectorizer(object):

    def __init__(self):
        self.value_indices = dict()
        self.labels = []
        self.max_idx = 0

    def normalize(self, x):
        return [x]

    def fit(self, data):
        for row in data:
            for n in self.normalize(row):
                if n not in self.value_indices:
                    self.value_indices[n] = self.max_idx
                    self.labels.append(n)
                    self.max_idx += 1

    def transform(self, value):
        ret = np.zeros((1, self.max_idx), dtype=np.int32)
        for n in self.normalize(value):
            idx = self.value_indices[n]
            ret[0, idx] = 1
        return ret


class TextVectorizer(BinaryVectorizer):

    def __init__(self):

        super(TextVectorizer, self).__init__()
        self.word_indices = dict()
        self.max_idx = 0

    def normalize(self, value):
        if value is None:
            return ['']

        value = value.strip()
        if value == '':
            return ['']

        words = remove_punc_keep_hyph_tokenizer.tokenize(value.lower())
        words = [i for i in words if i not in sw]
        return words

    # noinspection PyMethodOverriding
    def fit(self, data):
        return super(TextVectorizer, self).fit(data)

    # noinspection PyMethodOverriding
    def transform(self, value):
        return super(TextVectorizer, self).transform(value)
