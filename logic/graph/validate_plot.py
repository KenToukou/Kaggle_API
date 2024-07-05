import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AdversalValidateGraph:
    def plot_feature_importance(self, df: pd.DataFrame, feature_importances):
        indices = np.argsort(feature_importances)[::-1]
        sorted_feature_names = [df.columns[i] for i in indices]
        sorted_importances = feature_importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(range(len(sorted_importances)), sorted_importances, align="center")

        ax.set_xticks(range(len(sorted_importances)))
        ax.set_xticklabels(sorted_feature_names, rotation=90)

        ax.set_title("Feature Importances")
        ax.set_xlim([-1, len(sorted_importances)])
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")

        plt.tight_layout()
        plt.show()

    def plot_validate_train_test_distribution(self, model):
        lgb.plot_metric(model.evals_result_, metric="auc")
        plt.show()
