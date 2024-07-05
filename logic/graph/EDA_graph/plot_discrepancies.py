import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm


class Compare:
    @classmethod
    def _histogram(
        cls, train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list, hue
    ):
        n_columns = len(columns)
        fig, ax = plt.subplots(n_columns, 1, figsize=(10, 3 * n_columns))
        if n_columns == 1:
            ax = [ax]
        for i, column in tqdm(enumerate(columns)):
            sns.histplot(
                train_df[column],
                alpha=0.3,
                label="Train",
                hue=train_df[hue] if hue else None,
                ax=ax[i],
            )
            sns.histplot(
                test_df[column],
                alpha=0.3,
                label="Test",
                hue=test_df[hue] if hue else None,
                ax=ax[i],
            )
            ax[i].set_title(f"Histogram of {column}")
            ax[i].legend()

        plt.tight_layout()
        plt.show()

    @classmethod
    def _boxplot(
        cls, train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list, hue
    ):
        n_columns = len(columns)
        fig, axs = plt.subplots(n_columns, 2, figsize=(10, 5 * n_columns))
        if n_columns == 1:
            axs = axs.reshape(1, -1)

        for i, column in enumerate(columns):
            sns.boxplot(
                x=train_df[column],
                orient="v",
                ax=axs[i, 0],
                hue=train_df[hue] if hue else None,
            )
            sns.boxplot(
                x=test_df[column],
                orient="v",
                ax=axs[i, 1],
                hue=test_df[hue] if hue else None,
            )
            axs[i, 0].set_title(f"Train {column}")
            axs[i, 1].set_title(f"Test {column}")

            # Set a common title for the pair of plots
            fig.add_subplot(111, frame_on=False)
            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )
            plt.title(f"Boxplot of {column}", fontsize=16, pad=20)

        plt.tight_layout()
        plt.show()
