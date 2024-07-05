import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


class ColumnStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scalers = {}

    def fit(self, X, y=None):
        for col in self.columns:
            scaler = StandardScaler()
            scaler.fit(X[[col]])
            self.scalers[col] = scaler
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, scaler in self.scalers.items():
            X_copy[col] = scaler.transform(X[[col]])
        return X_copy


class ColumnMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scalers = {}

    def fit(self, X, y=None):
        for col in self.columns:
            scaler = MinMaxScaler()
            scaler.fit(X[[col]])
            self.scalers[col] = scaler
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, scaler in self.scalers.items():
            X_copy[col] = scaler.transform(X[[col]])
        return X_copy


class ColumnYeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, is_saved=False):
        self.columns = columns
        self.is_saved = is_saved
        self.transformers = {}

    def fit(self, X, y=None):
        for col in self.columns:
            transformer = PowerTransformer(method="yeo-johnson")
            transformer.fit(X[[col]])
            self.transformers[col] = transformer
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, transformer in self.transformers.items():
            X_copy[col] = transformer.transform(X[[col]])
            if self.is_saved:
                print("Origin")
                X_copy[f"origin_{col}"] = X[col]
        return X_copy


class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_clip=None, clip_lower=None, clip_upper=None):
        self.cols_to_clip = cols_to_clip
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if self.cols_to_clip is None:
            self.cols_to_clip = X.columns.tolist()

        for col in self.cols_to_clip:
            if self.clip_lower is not None:
                X_transformed[col] = np.clip(
                    X_transformed[col], self.clip_lower, self.clip_upper
                )

        return X_transformed
