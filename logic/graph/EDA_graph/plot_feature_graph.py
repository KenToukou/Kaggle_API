import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Description:
    @staticmethod
    def _scatter(
        df: pd.DataFrame, x_column: str, y_column: str, hue: str | None = None, ax=None
    ):
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)

    @classmethod
    def describe_some_columns_scatterplot(
        cls, df: pd.DataFrame, x_columns: list, y_columns: list, hue: str | None = None
    ):
        num_plots = len(x_columns) * len(y_columns)
        num_cols = min(3, len(x_columns))  # Adjust the number of columns for subplots
        num_rows = math.ceil(num_plots / num_cols)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
        )

        if num_rows * num_cols > 1:
            axes = axes.flatten()  # Flatten the axes array for easy iteration
        else:
            axes = [axes]  # Make it iterable

        plot_index = 0
        for x_col in x_columns:
            for y_col in y_columns:
                if plot_index < len(axes):
                    ax = axes[plot_index]
                    cls._scatter(df=df, x_column=x_col, y_column=y_col, hue=hue, ax=ax)
                    ax.set_title(f"{x_col} vs {y_col}")
                    plot_index += 1

        # Hide any unused subplots
        for i in range(plot_index, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    @classmethod
    def _correlation(cls, df: pd.DataFrame):
        # 相関係数の計算
        correlation_matrix = df.corr()

        # ヒートマップの描画
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,  # 相関係数をセルに表示
            cmap="viridis",  # カラーマップ
            vmin=-1,
            vmax=1,  # 相関係数の最小値と最大値
            linewidths=0.5,  # セルの境界線の太さ
        )
        plt.title("Correlation Matrix")
        plt.show()

    @classmethod
    def _bar(cls, df, col, hue):
        # Prepare the data for a stacked bar chart
        if hue:
            data = df.groupby([col, hue]).size().unstack(fill_value=0)
            data.plot(kind="bar", stacked=True, figsize=(10, 6))
        else:
            data = df[col].value_counts()
            data.plot(kind="bar", figsize=(10, 6))

        plt.title(f"Stacked Bar Chart of {col}" + (f" by {hue}" if hue else ""))
        plt.ylabel("Count")
        plt.show()
