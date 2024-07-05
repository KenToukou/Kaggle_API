import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelRegressionEvaluate:
    def __init__(self, predict_data: list, true_data: list):
        self._predict = predict_data
        self._true = true_data
        if len(self._predict) != len(self._true):
            raise ValueError("長さが違う！！！！")

    def valid_mean_absolute_error(self):
        self.mae_score = mean_absolute_error(self._true, self._predict)
        # print(f"MAE:{self.mae_score:.2}")
        return self.mae_score

    def valid_mean_squared_error(self):
        self.mse_score = mean_squared_error(self._true, self._predict)
        # print(f"MSE:{self.mse_score:.2}")
        return self.mse_score

    def valid_root_mean_squared_error(self):
        self.rmse_score = np.sqrt(mean_squared_error(self._true, self._predict))
        # print(f"RMSE:{self.rmse_score:.2}")
        return self.rmse_score

    def valid_R2_squared(self):
        self._R2 = r2_score(self._true, self._predict)
        # print(f"R2(当てはまりやすさ):{self._R2:.2}")
        return self._R2

    def create_evaluate_df(self):
        self.valid_mean_absolute_error()
        self.valid_mean_squared_error()
        self.valid_root_mean_squared_error()
        self.valid_R2_squared()
        df = pd.DataFrame(
            {
                "MAE": [self.mae_score],
                "MSE": [self.mse_score],
                "RMSE": [self.rmse_score],
                "R2": [self._R2],
            },
            index=[0],
        )
        return df


class ModelClassificationEvaluate:
    def __init__(self, predict_data: list, true_data: list, average="binary"):
        self._predict = predict_data
        self._true = true_data
        self._average = average
        if len(self._predict) != len(self._true):
            raise ValueError("長さが違う！！！！")

    def valid_accuracy_score(self):
        self._acc_score = accuracy_score(self._true, self._predict)
        # print(f"正解率:{self._acc_score:.2}")
        return self._acc_score

    def valid_precision(self):
        self._precision = precision_score(
            self._true, self._predict, average=self._average
        )
        # print(f"適合率:{self._precision:.2}")
        return self._precision

    def valid_recall(self):
        self._recall = recall_score(self._true, self._predict, average=self._average)
        # print(f"再現率:{self._recall:.2}")
        return self._recall

    def valid_f_value(self):
        self._f = f1_score(self._true, self._predict, average=self._average)
        # print(f"F値:{self._f:.2}")
        return self._f

    def auc_and_roc_plot(self, true_proba, predict_proba):
        fpr, tpr, thresholds = roc_curve(true_proba, predict_proba)
        auc_score = auc(fpr, tpr)
        print(f"AUC(面積):{auc_score:.2}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label="Logistic Regression")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    @classmethod
    def roc_auc_value(self, true_proba, predict_proba):
        auc_score = roc_auc_score(true_proba, predict_proba)

        # print(f"AUC(面積):{auc_score:.2}")
        return auc_score

    def create_evaluate_df(self):
        self.valid_accuracy_score()
        self.valid_precision()
        self.valid_recall()
        self.valid_f_value()
        df = pd.DataFrame(
            {
                "正解率": [self._acc_score],
                "適合率": [self._precision],
                "再現率": [self._recall],
                "F値": [self._f],
            },
            index=[0],
        )
        # print(df)
        return df
