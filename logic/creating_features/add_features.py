import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(
            f"The process of dropping columns has been completed successfully.\n{self.columns}"
        )
        return X.drop(self.columns, axis=1)


class AddSumColumns(BaseEstimator, TransformerMixin):
    def __init__(self, add_columns):
        self.add_columns = add_columns  # 列のリストを初期化

    def fit(self, X, y=None):
        # このトランスフォーマーはfitメソッドで学習する必要がないので、自分自身を返す
        return self

    def transform(self, X):
        # 指定された列の合計を計算し、新しい列として追加
        features_name = "sum"
        for name in self.add_columns:
            features_name += f"_{name}"

        X[features_name] = X[self.add_columns].sum(axis=1)

        return X


class AddMultipleColumns(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2):
        self.col1 = col1
        self.col2 = col2

    def fit(self, X, y=None):
        # このトランスフォーマーはfitメソッドで学習する必要がないので、自分自身を返す
        return self

    def transform(self, X):
        # 指定された列の合計を計算し、新しい列として追加

        X[f"multiple_{self.col1}_{self.col2}"] = X[self.col1] * X[self.col2]

        return X


class AddDiffColumns(BaseEstimator, TransformerMixin):
    def __init__(self, column1, column2):
        self.column1 = column1  # 列のリストを初期化
        self.column2 = column2  # 列のリストを初期化

    def fit(self, X, y=None):
        # このトランスフォーマーはfitメソッドで学習する必要がないので、自分自身を返す
        return self

    def transform(self, X):

        # 指定された列の合計を計算し、新しい列として追加
        X[f"diff_{self.column1}-{self.column2}"] = X[self.column1] - X[self.column2]

        return X


class AddGroupbyAverageColumn(BaseEstimator, TransformerMixin):
    """_summary_
    f"{column}_average_by_{self.groupby_column}"
    """

    def __init__(self, groupby_column, average_columns):
        self.groupby_column = (
            groupby_column  # ここは単数形か複数形を正確にする必要がある
        )
        self.average_columns = average_columns

    def fit(self, X, y=None):
        self.groupby_means_ = X.groupby(self.groupby_column)[
            self.average_columns
        ].mean()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.average_columns:
            new_column_name = f"{column}_average_by_{self.groupby_column}"
            X_copy = X_copy.merge(
                self.groupby_means_[column]
                .reset_index()
                .rename({column: new_column_name}, axis=1),
                on=self.groupby_column,
                how="left",
            )

        return X_copy


class AddGroupbyMedianColumn(BaseEstimator, TransformerMixin):
    """_summary_
    f"{column}_median_by_{self.groupby_column}"
    """

    def __init__(self, groupby_column, average_columns):
        self.groupby_column = (
            groupby_column  # ここは単数形か複数形を正確にする必要がある
        )
        self.average_columns = average_columns

    def fit(self, X, y=None):
        self.groupby_medians_ = X.groupby(self.groupby_column)[
            self.average_columns
        ].median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.average_columns:
            new_column_name = f"{column}_median_by_{self.groupby_column}"
            X_copy = X_copy.merge(
                self.groupby_medians_[column]
                .reset_index()
                .rename({column: new_column_name}, axis=1),
                on=self.groupby_column,
                how="left",
            )

        return X_copy


class AddBinningColumn(BaseEstimator, TransformerMixin):
    """_summary_
    f"{self.target_column}_binned"
    """

    def __init__(self, target_column: str, bin_edges: list, labels=None):
        self.bin_edges = bin_edges
        self.target_column = target_column
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # ビン分割を行い、新しいカラムを追加
        if self.labels:
            X_copy[f"{self.target_column}_binned"] = pd.cut(
                X_copy[self.target_column], bins=self.bin_edges, labels=self.labels
            )
        else:
            X_copy[f"{self.target_column}_binned"] = pd.cut(
                X_copy[self.target_column], bins=self.bin_edges, labels=False
            )
        return X_copy


class AddQcutColumn(BaseEstimator, TransformerMixin):

    def __init__(self, target_columns: list, bin_num: int, labels=False):
        self.target_columns = target_columns  # 分割する列のリスト
        self.labels = labels  # ビンのラベル
        self.bin_num = bin_num  # ビンの数

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 入力XをDataFrameとして扱う
        X = pd.DataFrame(X)

        # 指定された各列に対して操作を行う
        for col in self.target_columns:
            # qcutを使用してデータを等間隔に分割し、新しい列として追加
            col_name = f"{col}_qcut"
            X[col_name] = pd.qcut(X[col], self.bin_num, labels=self.labels)

        return X
