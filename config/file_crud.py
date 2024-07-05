import logging
import pickle
from io import BytesIO
from pathlib import Path

import pandas as pd

from .base import CsvFileABC, ExcelFileABC, FolderABC, PklFileABC, Singleton

"""
注意項目
1 upload: ファイル名だけで良い(拡張子を引数に含めなくてOK)
2 get : 拡張子も含む必要あり。
"""


class PklFile(PklFileABC, Singleton):
    def get(self, path):
        return self._read(path)

    def upload(self, path, dic):
        self._create(path, dic)

    def _create(self, path, dic):
        pickle.dump(dic, open(path, "wb"))

    def _delete(self):
        return super()._delete()

    def _read(self, path):
        try:
            return pickle.load(open(path, "rb"))
        except:  # noqa
            return False

    def _upload(self):
        return super()._upload()


class CsvFile(CsvFileABC, Singleton):
    def get(self, file_path, usecols, header):

        return self._read(file_path, use_cols=usecols, header=header)

    def upload(self, df: pd.DataFrame, file_path, file_name):
        return self._upload(df=df, file_path=file_path, file_name=file_name)

    def _read(self, path, use_cols, header):
        try:
            return pd.read_csv(path, usecols=use_cols, header=header)
        except:  # noqa
            print(f"ファイルは存在しません。\n{path}")
            return False

    def _upload(self, df: pd.DataFrame, file_path, file_name):
        df.to_csv(f"{file_path}/{file_name}.csv", index=False)

    def _delete(self):
        pass

    def _create(self):
        pass


class ExcelFile(ExcelFileABC, Singleton):

    def read(self, excel_file_path, sheet_name, use_cols, header):
        return self._read(excel_file_path, sheet_name, use_cols, header)

    def _read(self, excel_file_path, sheet, cols, header):
        return pd.read_excel(
            io=excel_file_path, sheet_name=sheet, usecols=cols, header=header
        )

    def _create(self, dict_df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for sheet_name, df in dict_df.items():
                df.to_excel(writer, sheet_name=sheet_name)
                logging.info("最終フェーズ: %s の書き込み完了.", sheet_name)
                logging.info("----------------------------------------------------")
        excel_data = output.getvalue()
        logging.info("<<<<<<<<<<<<<<<<書き込み完了>>>>>>>>>>>>>>>>>>>")
        return excel_data

    def _upload(self, excel_data, path, name):
        file_path = path / f"{name}.xlsx"
        with open(file_path, "wb") as file:
            file.write(excel_data)

    def _delete(self):
        pass


class Folder(FolderABC, Singleton):
    def get(self, path: Path):
        return self._read(path)

    def create(self, path, name):
        return self._create(path, name)

    def _read(self, path: Path):
        return list(path.glob("*"))

    def _create(self, path, name):
        output_dir = path / name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _delete(self, path: Path):
        pass
