import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # One-Hot Encodingを適用するカラム
        self.encoders = {}  # 各カラムに対するエンコーダーを保存

    def fit(self, X: pd.DataFrame, y=None):
        # 指定されたカラムに対してOne-Hot Encoderをfitする
        for column in self.columns:
            encoder = OneHotEncoder(sparse_output=False)
            self.encoders[column] = encoder.fit(X[[column]])
        return self

    def transform(self, X):
        # One-Hot Encodingを適用
        X_copy = X.copy()
        for column in self.columns:
            encoder = self.encoders[column]
            encoded = encoder.transform(X[[column]])

            # One-Hot Encodingの結果をデータフレームに追加
            for i, category in enumerate(encoder.categories_[0]):
                new_column_name = f"{column}_{category}"
                X_copy[new_column_name] = encoded[:, i]

            # 元のカラムを削除
            X_copy.drop(column, axis=1, inplace=True)

        return X_copy


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # Columns to apply Label Encoding
        self.encoders = {}  # Dictionary to store encoders for each column

    def fit(self, X, y=None):
        # Fit Label Encoder for specified columns
        for column in self.columns:
            encoder = LabelEncoder()
            # Handle NaN by converting type to str
            self.encoders[column] = encoder.fit(X[column].astype(str))
        return self

    def transform(self, X):
        # Apply Label Encoding
        X_copy = X.copy()
        for column in self.columns:
            encoder = self.encoders[column]
            # Handle NaN by converting type to str
            X_copy[f"label_{column}"] = encoder.transform(X_copy[column].astype(str))
        return X_copy


class CustomFrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # Frequency Encodingを適用するカラム
        self.freq_maps = {}  # 各カラムの頻度マップを保存

    def fit(self, X, y=None):
        # 指定されたカラムの各値の出現頻度を計算する
        for column in self.columns:
            frequency = X[column].value_counts(normalize=True)  # 頻度を割合で計算
            self.freq_maps[column] = frequency.to_dict()
        return self

    def transform(self, X):
        # Frequency Encodingを適用
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].map(self.freq_maps[column]).fillna(0)
        return X_copy


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, target_name, n_splits=5):
        self.columns = columns  # Target Encodingを適用するカラムのリスト
        self.target_name = target_name  # ターゲットとなるカラム
        self.n_splits = n_splits  # K-foldの分割数
        self.mapping = {}

    def fit(self, X, y=None):
        # ターゲット変数との関係に基づいてマッピングを学習する
        for column in self.columns:
            mean_of_target = X.groupby(column)[self.target_name].mean()
            self.mapping[column] = mean_of_target.to_dict()
        return self

    def transform(self, X, y=None):
        # K-foldを用いてTarget Encodingを適用
        X_copy = X.copy()
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for column in self.columns:
            X_copy[f"target_encoded_{column}"] = (
                np.nan
            )  # エンコーディング結果を格納する列を追加

            for train_index, val_index in kfold.split(X):
                train, val = X.iloc[train_index], X.iloc[val_index]
                val[f"target_encoded_{column}"] = val[column].map(
                    train.groupby(column)[self.target_name].mean()
                )
                X_copy.iloc[
                    val_index, X_copy.columns.get_loc(f"target_encoded_{column}")
                ] = val[f"target_encoded_{column}"]

            # K-foldで計算されなかった値を全体平均で埋める
            X_copy[f"encoded_{column}"].fillna(self.mapping[column], inplace=True)

        return X_copy
