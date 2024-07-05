import numpy as np
import pandas as pd
from prettytable import PrettyTable


class DataInfo:
    def __init__(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_columns: list
    ):
        self.target_columns = target_columns
        self._train_df = train_df
        self.test_df = test_df
        self.numerical_columns = self.numerical_columns = self._train_df.select_dtypes(
            include=["number"]
        ).columns.to_list()
        self.categorical_columns = self._train_df.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

    @property
    def target_df(self):
        if self._train_df[self.target_columns].min().min() < 1:
            print("既に変換済み")
            return self._train_df[self.target_columns]
        else:
            return np.log(self._train_df[self.target_columns])

    @property
    def total_df(self):
        pre_total_df = pd.concat([self.train_df, self.test_df], axis=0).reset_index(
            drop=True
        )
        if all(column in pre_total_df.columns for column in self.target_columns):
            return pre_total_df.drop(self.target_columns, axis=1)
        else:
            return pre_total_df

    @property
    def train_df(self):
        return self._train_df.drop(self.target_columns, axis=1)

    @property
    def EDA_df(self):
        return self._train_df

    @classmethod
    def show_status(self, df: pd.DataFrame):
        table = PrettyTable(["Variable", "Missing Values", "Duplicates", "Outliers"])

        for col in df.columns:
            missing_values = df[col].isnull().sum()

            duplicates = df.duplicated(subset=[col]).sum()
            if df[col].dtype in ["int64", "float64"]:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] - mean).abs() > 3 * std).sum()
            else:
                outliers = "N/A"

            table.add_row([col, missing_values, duplicates, outliers])
        print(table)

    @classmethod
    def has_null_values(df: pd.DataFrame, columns: list):
        dict_null_columns = {}
        for col in columns:
            missing_values = df[col].isnull().sum()
            if missing_values != 0:
                dict_null_columns[col] = missing_values
        return dict_null_columns
