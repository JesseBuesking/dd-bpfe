

from collections import Counter
import re
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from bpfe.entities import Data
from bpfe.feature_engineering import all_ngrams_list
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


sbs = SnowballStemmer('english')
english_stopwords = stopwords.words('english')


class BPFEVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, ngrams=1, use_stemmer=False, use_grades=False,
                 use_titles=False, filter_stopwords=True, binarize=False,
                 min_df=None, max_df=None):
        self.dv = None
        self.ngrams = ngrams
        self.use_stemmer = use_stemmer
        self.use_grades = use_grades
        self.use_titles = use_titles
        self.filter_stopwords = filter_stopwords
        self.binarize = binarize
        self.min_df = min_df
        self.max_df = max_df
        self.counts = Counter()

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        return self.fit_transform(X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        tmp = []
        for data in X:
            feats = self.features(data)
            tmp.append(feats)

        for d in tmp:
            for k in d.keys():
                self.counts[k] += 1

        if self.min_df is not None:
            for i in range(len(tmp)):
                d = tmp[i]
                for k in d.keys():
                    if self.counts[k] < self.min_df:
                        del d[k]

        if self.max_df is not None:
            for i in range(len(tmp)):
                d = tmp[i]
                for k in d.keys():
                    if self.counts[k] > self.max_df:
                        del d[k]

        if self.dv is None:
            self.dv = DictVectorizer().fit(tmp)

        ft = self.dv.transform(tmp)
        return ft

    def features(self, data):
        feats = self.word_features(data)

        if self.use_grades:
            feats.update(self.grade_features(data))
        if self.use_titles:
            feats.update(self.title_features(data))

        return feats

    def grade_features(self, data):
        d = dict()
        for idx, grade in enumerate(data.grades):
            if grade:
                d['grade - {}'.format(idx)] = 1
        return d

    def title_features(self, data):
        d = dict()
        for idx, title in enumerate(data.title):
            if title:
                d['title - {}'.format(idx)] = 1
        return d

    def word_features(self, data):
        d = dict()
        for attr in Data.text_attributes:
            value = data.cleaned[attr + '-mapped']
            # value = getattr(data, attr)
            b_o_w = []
            for i in self.bow(value):
                if self.use_stemmer:
                    i = sbs.stem(i)

                b_o_w.append(i)

            ng = all_ngrams_list(b_o_w, self.ngrams)
            if self.filter_stopwords:
                # trim the stopwords from either end
                # e.g. salaries and wages -> salaries and wages
                #      salaries and -> salaries
                new_ng = []
                for i in ng:
                    while len(i) > 0 and i[0] in english_stopwords:
                        i = i[1:]
                    while len(i) > 0 and i[-1] in english_stopwords:
                        i = i[:-1]
                    if len(i) > 0:
                        new_ng.append(tuple(i))

                # this helps a lot apparently
                for i in range(len(new_ng)):
                    pre = []
                    suf = []
                    for j in new_ng[i]:
                        pre.append(j[:3])
                        suf.append(j[-3:])
                    new_ng += [tuple(pre)]
                    new_ng += [tuple(suf)]

                # only keep the distinct occurrences
                ng = list(set(new_ng))
            else:
                ng = [tuple(i) for i in ng]

            if self.binarize:
                for i in ng:
                    d[i] = 1
                    d[('|' + attr + '|',) + i] = 1
            else:
                for i in ng:
                    d[i] = d.get(i, 0) + 1
                    d[('|' + attr + '|',) + i] = d.get(i, 0) + 1

        return d

    def bow(self, string):
        # return util.sentence_splitter(string)
        for word in re.findall(
                r'GRADE=k\|k|GRADE=k\|\d+|GRADE=\d+\|\d+|TITLE=\d+|\w+|\d+',
                string
        ):
            if 'GRADE=' in word or 'TITLE=' in word:
                continue
            yield word
