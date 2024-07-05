import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from config.set_logging import configure_logging
from config.utilities import OsControl
from logic.analysis import DataInfo
from logic.graph import EDA
from logic.models import AdversalValidator


class FeatureGenerator(OsControl, EDA):
    def __init__(self, target_columns: list):
        OsControl.__init__(self)
        self.target_columns: list = target_columns

    def set_data(self, folder_path):
        dict_datas = self.get_df_dict_datas(folder_path)
        self._train_df = dict_datas["train"]
        self._test_df = dict_datas["test"]
        self.data_info_class = DataInfo(
            train_df=self._train_df,
            test_df=self._test_df,
            target_columns=self.target_columns,
        )

        """ 初期のデータを保存する。 """
        self.target_df = self.data_info_class.target_df
        self.train_df = self.data_info_class.train_df
        self.test_df = self.data_info_class.test_df
        self.total_df = self.data_info_class.total_df
        self.EDA_df = self.data_info_class.EDA_df

    def run_pipline(self, df: pd.DataFrame, pipline_dict: dict):
        self._pipline_dict = pipline_dict
        self._set_pipline(pipline_dict)
        self.total_df_arranged = self.pipline_class.fit_transform(df)
        self.train_df_arranged = self.total_df_arranged.loc[
            : self._train_df.shape[0] - 1
        ]
        self.test_df_arranged = self.total_df_arranged.loc[
            self._train_df.shape[0] : self.total_df_arranged.shape[0]
        ]
        print("Pipeline process completed successfully.")
        return self.total_df_arranged

    def _set_pipline(self, pipline_dict: dict):
        pipline_list = []
        for key, val in pipline_dict.items():
            pipline_list += [(key, val)]
        self.pipline_class = Pipeline(pipline_list)

    def _concat_target_columns_with_train_df(self):
        return pd.concat([self.train_df_arranged, self.target_df], axis=1)

    def save_log_info(self, folder_name):
        configure_logging(folder_name=folder_name)
        logging.info("start")
        logging.info("created_data_info")
        logging.info("-------------------------------------------------------")
        for key, value in self._pipline_dict.items():
            logging.info(f"{key} : {value}")
        self.create_folder(self.folder_path_model.home_data_dir, folder_name)
        self.upload_a_csv_file(
            df_csv=self._concat_target_columns_with_train_df(),
            output_file_path=f"{self.folder_path_model.home_data_dir}/{folder_name}",
            name="train",
        )
        self.upload_a_csv_file(
            df_csv=self.test_df_arranged,
            output_file_path=f"{self.folder_path_model.home_data_dir}/{folder_name}",
            name="test",
        )
        logging.info("-------------------------------------------------------")
        logging.info("DataFrameのカラム")
        logging.info(self.test_df_arranged.columns)

    def save_feedback_data(self, df, folder_name):
        self.create_folder(self.folder_path_model.feedback_data_home_dir, folder_name)
        self.upload_a_csv_file(
            df_csv=df,
            output_file_path=f"{self.folder_path_model.feedback_data_home_dir}/{folder_name}",
            name="feedback",
        )

    def adversal_check(self, train_data, test_data):
        adv_class = AdversalValidator()
        adv_class.set_data(train_data=train_data, test_data=test_data)
        adv_class.evaluate_features()

    def output_predicted_data(self, pred: list, target_column: str, path: Path):
        df = pd.read_csv(path)
        df[target_column] = pred
        return df

    def save_output_data(
        self, pred: list, folder_name: str, target_column: str, path: Path
    ):
        new_dir = self.create_folder(
            self.folder_path_model.output_path, new_dir_name=folder_name
        )
        new_df = self.output_predicted_data(pred, target_column, path)
        self.upload_a_csv_file(
            df_csv=new_df, output_file_path=new_dir, name="submission"
        )
        return new_df
