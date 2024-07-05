import pandas as pd

from .EDA_graph import Compare, Description


class EDA(Compare, Description):
    """
    Train と Testのデータの類似性の確認。
    """

    def compare_histgram(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, columns, hue=None
    ):
        return Compare._histogram(
            train_df=train_df, test_df=test_df, columns=columns, hue=hue
        )

    def compare_boxplot(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, columns, hue=None
    ):
        return Compare._boxplot(
            train_df=train_df, test_df=test_df, columns=columns, hue=hue
        )

    """
    Scatter
    """

    def describe_scatterplot(
        self,
        base_df: pd.DataFrame,
        x_column: str,
        y_column: str,
        hue: str | None = None,
    ):
        return Description._scatter(
            df=base_df, x_column=x_column, y_column=y_column, hue=hue
        )

    def describe_some_columns_scatterplot(
        self, df: pd.DataFrame, x_columns: list, y_columns: list, hue: str | None = None
    ):
        return Description.describe_some_columns_scatterplot(
            df=df, x_columns=x_columns, y_columns=y_columns, hue=hue
        )

    """
    Corr
    """

    def describe_correlation_heatmap(self, df):
        return Description._correlation(df=df)

    """
    Bar
    """

    def describe_bar(
        self,
        df: pd.DataFrame,
        col_x: str,
        hue: str | None = None,
    ):
        return Description._bar(df=df, col=col_x, hue=hue)
