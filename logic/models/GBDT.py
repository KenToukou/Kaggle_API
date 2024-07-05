import logging

import numpy as np
import xgboost as xgb

# import pandas as pd
from sklearn.metrics import mean_squared_log_error, roc_auc_score


class XGBModel(object):
    def __init__(self, param_dict: dict = {}):
        self.param_dict = param_dict
        self.pred_min = 1
        self.pred_max = 29

    def train(self, X_train, y_train, X_valid, y_valid):
        logging.info(
            f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}"
        )
        logging.info(
            f"Validation data shape: {X_valid.shape}, Validation labels shape: {y_valid.shape}"
        )
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        self.model = xgb.train(
            params=self.param_dict,
            dtrain=d_train,
            num_boost_round=1000,
            evals=[(d_valid, "validation")],
            feval=self.roc_auc,
            early_stopping_rounds=10,
            maximize=True,
        )

    def train_all_data(self, train, valid):
        d_train = xgb.DMatrix(train, label=valid)
        self.model = xgb.train(
            params=self.param_dict,
            dtrain=d_train,
            num_boost_round=1000,
        )

    def roc_auc(self, preds, dtrain):
        # Extract the true labels from the DMatrix
        y_true = dtrain.get_label()
        # Compute the ROC AUC
        roc_auc_value = roc_auc_score(y_true, preds)
        return "roc_auc", roc_auc_value

    def rmsle(self, preds, dtrain):
        # Extract the true labels from the DMatrix
        y_true = dtrain.get_label()
        # Compute the RMSLE
        rmsle_value = mean_squared_log_error(y_true, preds)
        return "rmsle", rmsle_value

    def predict(self, X):
        d_test = xgb.DMatrix(X)
        pred = self.model.predict(d_test)
        return pred

    def _clip_pred_value(self, pred):
        pred_clipped = np.clip(pred, a_min=self.pred_min, a_max=self.pred_max)
        return pred_clipped

    def get_feature_importance(self):
        feature_importances = self.model.get_score(importance_type="weight")
        return feature_importances
