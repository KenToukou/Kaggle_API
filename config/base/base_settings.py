import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class PklFileABC(metaclass=ABCMeta):
    @abstractmethod
    def _create(self):
        pass

    @abstractmethod
    def _upload(self):
        pass

    @abstractmethod
    def _read(self):
        pass

    @abstractmethod
    def _delete(self):
        pass


class CsvFileABC(metaclass=ABCMeta):
    @abstractmethod
    def _create(self):
        pass

    @abstractmethod
    def _upload(self):
        pass

    @abstractmethod
    def _read(self):
        pass

    @abstractmethod
    def _delete(self):
        pass


class ExcelFileABC(metaclass=ABCMeta):
    def upload(self, dict_df, path: Path, name: str):
        excel_data = self._create(dict_df)
        self._upload(excel_data, path, name)
        logging.info(f"Excel data has been saved to '{path}/{name}'.xlsx")

    @abstractmethod
    def _create(self, dict_df):
        pass

    @abstractmethod
    def _upload(self, excel_data, path: Path, name: str):
        pass

    @abstractmethod
    def _read(self):
        pass

    @abstractmethod
    def _delete(self):
        pass


class FolderABC(metaclass=ABCMeta):
    @abstractmethod
    def _create(self, path: Path, name: str):
        pass

    @abstractmethod
    def _read(self, path: Path):
        pass

    @abstractmethod
    def _delete(self, path: Path):
        pass
