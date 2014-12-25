

from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):

    def transform(self, X, **_):
        return X.toarray()

    # noinspection PyUnusedLocal
    def fit(self, X, y, **_):
        return self

    def get_params(self, **_):
        return dict()
