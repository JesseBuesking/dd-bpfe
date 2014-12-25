

from sklearn.base import TransformerMixin


class TitleFeatures(TransformerMixin):

    def transform(self, X, **_):
        ret = []
        for row in X:
            ret.append([row.is_title] + row.title)
        return ret

    # noinspection PyUnusedLocal
    def fit(self, X, y, **_):
        return self
