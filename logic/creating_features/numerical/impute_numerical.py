import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AverageImputer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df: pd.DataFrame, cols):
        self._df = df
        for column in df.columns:
            if column in cols:
                df[column] = df[column].fillna(df[column].mean())
        return df


class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df: pd.DataFrame, cols):
        self._df = df
        for column in df.columns:
            if column in cols:
                df[column] = df[column].fillna(df[column].median())
        return df


class QcutAverageImputer(BaseEstimator, TransformerMixin):
    def __init__(self, qcut_column, target_column, q=4):
        self.qcut_column = qcut_column  # 区分するためのカラム
        self.target_column = target_column  # 埋める対象のカラム
        self.q = q  # 区分の数

    def fit(self, X, y=None):
        # qcut で区分を作成し、各区分の平均値を計算
        self.bins = pd.qcut(X[self.qcut_column], self.q, duplicates="drop")
        self.means = X.groupby(self.bins)[self.target_column].mean()
        return self

    def transform(self, X):
        # 区分ごとに欠損値を埋める
        X_copy = X.copy()
        for bin in self.means.index:
            mean = self.means[bin]
            mask = (X_copy[self.qcut_column] >= bin.left) & (
                X_copy[self.qcut_column] <= bin.right
            )
            X_copy.loc[
                mask & X_copy[self.target_column].isnull(), self.target_column
            ] = mean
        return X_copy


class QcutMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, qcut_column, target_column, q=4):
        self.qcut_column = qcut_column  # 区分するためのカラム
        self.target_column = target_column  # 埋める対象のカラム
        self.q = q  # 区分の数

    def fit(self, X, y=None):
        # qcut で区分を作成し、各区分の平均値を計算
        self.bins = pd.qcut(X[self.qcut_column], self.q, duplicates="drop")
        self.means = X.groupby(self.bins)[self.target_column].median()
        return self

    def transform(self, X):
        # 区分ごとに欠損値を埋める
        X_copy = X.copy()
        for bin in self.means.index:
            median = self.means[bin]
            mask = (X_copy[self.qcut_column] >= bin.left) & (
                X_copy[self.qcut_column] <= bin.right
            )
            X_copy.loc[
                mask & X_copy[self.target_column].isnull(), self.target_column
            ] = median
        return X_copy


class GroupByMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, groups, target_column, q=4):
        self.groups = groups
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y=None):
        # 中央値を計算
        self.median_values = X.groupby(by=self.groups)[self.target_column].median()
        return self

    def transform(self, X):
        # グループごとに欠損値を埋める
        X_copy = X.copy()
        for group in self.median_values.index:
            median = self.median_values.loc[group]
            mask = X_copy[self.groups] == group
            X_copy.loc[
                mask & X_copy[self.target_column].isnull(), self.target_column
            ] = median
        return X_copy


class GroupByAverageImputer(BaseEstimator, TransformerMixin):
    def __init__(self, groups, target_column, q=4):
        self.groups = groups
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y=None):
        self.average_values = X.groupby(by=self.groups)[self.target_column].average()
        return self

    def transform(self, X):
        # グループごとに欠損値を埋める
        X_copy = X.copy()
        for group in self.average_values.index:
            average = self.average_values.loc[group]
            mask = X_copy[self.groups] == group
            X_copy.loc[
                mask & X_copy[self.target_column].isnull(), self.target_column
            ] = average
        return X_copy


class MissingValueIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.cols:
            new_col_name = f"{col}_is_null"
            df[new_col_name] = df[col].isnull().astype(int)
        return df
