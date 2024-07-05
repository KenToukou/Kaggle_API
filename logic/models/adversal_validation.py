import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from logic.graph import AdversalValidateGraph


class AdversalValidator(object):
    def __init__(self):
        self.model = lgb.LGBMClassifier(n_estimators=1000, random_state=42)

    def set_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        combined_data["is_test"] = 0
        combined_data.loc[train_data.shape[0] :, "is_test"] = 1
        train_adv, valid_adv = train_test_split(
            combined_data, test_size=0.33, random_state=42, shuffle=True
        )
        self.X_train_adv = train_adv.drop(["is_test"], axis=1)
        self.y_train_adv = train_adv["is_test"]
        self.X_valid_adv = valid_adv.drop(["is_test"], axis=1)
        self.y_valid_adv = valid_adv["is_test"]

    def _learn(self):
        self.model.fit(
            self.X_train_adv,
            self.y_train_adv,
            eval_set=[
                (self.X_train_adv, self.y_train_adv),
                (self.X_valid_adv, self.y_valid_adv),
            ],
            eval_names=["train", "valid"],
            eval_metric="auc",
            force_col_wise=True,
        )

    def evaluate_features(self):
        validate_graph_class = AdversalValidateGraph()
        self._learn()
        feature_importances = self.model.feature_importances_
        validate_graph_class.plot_validate_train_test_distribution(self.model)
        validate_graph_class.plot_feature_importance(
            self.X_train_adv, feature_importances
        )
