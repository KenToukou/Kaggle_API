"""各コンペごとに使う特徴量を作成する。"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OriginalFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, generator_dict: dict):
        self.generator_dict: dict = generator_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        for key, method in self.generator_dict.items():
            X[key] = X.apply(method, axis=1)
        return X
