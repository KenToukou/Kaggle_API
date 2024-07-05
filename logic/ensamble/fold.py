import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from config import configure_logging_score
from logic.evaluation.evaluate import ModelClassificationEvaluate


class CrossValidation:
    def __init__(self, model, train, target, folder_name):
        self.model = model
        self.train = train
        self.target = (
            target  # validをtargetに変更して、これが目的変数であることを明確に
        )
        configure_logging_score(folder_name)
        self.scores: list = []
        self.models_list = []
        self.feature_importances: list = []

    def learn(self, n_split: int):
        kf = KFold(n_splits=n_split, shuffle=True, random_state=7)
        logging.info("Param:")
        logging.info(self.model.param_dict)
        logging.info("+++++++++++++++++++++++++++++++++++++++++")
        validation_dfs = []
        for tr_idx, va_idx in kf.split(
            self.train
        ):  # self.valid の代わりに self.train を使用
            tr_x, va_x = self.train.iloc[tr_idx], self.train.iloc[va_idx]
            tr_y, va_y = self.target.iloc[tr_idx], self.target.iloc[va_idx]
            self.model.train(tr_x, tr_y, va_x, va_y)
            pred = self.model.predict(va_x)
            score = ModelClassificationEvaluate.roc_auc_value(
                true_proba=va_y, predict_proba=pred
            )
            logging.info(f"auc_roc:{score}")
            self.scores.append(score)
            self.models_list.append(self.model)
            logging.info("Importance features:")
            logging.info(self.model.get_feature_importance())
            self.feature_importances.append(self.model.get_feature_importance())

            va_y_1d = va_y.iloc[:, 0] if va_y.ndim > 1 else va_y
            difference = [
                (np.log((1 + y) / (1 + p)) ** 2) for p, y in zip(pred, va_y_1d)
            ]
            va_x["prediction_difference"] = difference

            validation_dfs.append(va_x)
            logging.info("=================================================")
        logging.info(f"Finally:{np.mean(self.scores)}")

        all_validations_df = pd.concat(validation_dfs, ignore_index=True)
        return all_validations_df

    def predict(self, test):
        preds = []
        for model in self.models_list:
            pred = model.predict(test)
            preds.append(pred)
        preds_array = np.array(preds)
        mean_values = np.mean(preds_array, axis=0)
        return mean_values


class StraitifiedValidation:
    def __init__(self, model, train, valid, folder_name):
        self.model = model
        self.train = train
        self.valid = valid
        configure_logging_score(folder_name)
        self.scores: list = []
        self.models_list = []
        self.feature_importances: list = []

    def learn(self, n_split: int):
        kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=7)
        logging.info("Param:")
        logging.info(self.model.param_dict)
        logging.info("+++++++++++++++++++++++++++++++++++++++++")
        validation_dfs = []  # 予測が一致するかどうかの情報を格納するためのリスト
        for tr_idx, va_idx in kf.split(self.train, self.valid):
            tr_x, va_x = self.train.iloc[tr_idx], self.train.iloc[va_idx]
            tr_y, va_y = self.valid.iloc[tr_idx], self.valid.iloc[va_idx]
            self.model.train(tr_x, tr_y, va_x, va_y)
            pred = self.model.predict(va_x)
            score = mean_squared_error(y_pred=pred, y_true=va_y)
            logging.info(f"mean_squared_error:{score}")
            self.scores.append(score)
            self.models_list.append(self.model)
            logging.info("Importance features:")
            logging.info(self.model.get_feature_importance())
            self.feature_importances.append(self.model.get_feature_importance())

            # va_y を1次元配列に変換する
            va_y_1d = va_y.iloc[:, 0]
            # 予測値と実際の値の差異を示すカラムを追加する
            difference = [abs(p - y) for p, y in zip(pred, va_y_1d)]
            va_x["prediction_difference"] = difference

            validation_dfs.append(va_x)
            logging.info("=================================================")
        logging.info(f"Finally:{np.mean(self.scores)}")

        # すべてのfoldの検証結果を結合
        all_validations_df = pd.concat(validation_dfs, ignore_index=True)
        return all_validations_df

    def predict(self, test):
        preds = []
        for model in self.models_list:
            pred = model.predict(test)
            preds.append(pred)
        preds_array = np.array(preds)
        mean_values = np.mean(preds_array, axis=0)
        return mean_values
