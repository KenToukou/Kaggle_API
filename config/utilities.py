from schemas import FolderPathModel

from .file_crud import CsvFile, ExcelFile, Folder, PklFile  # noqa


class OsControl:
    def __init__(self):
        self.folder_path_model = FolderPathModel()

        self._csv_repo: CsvFile = CsvFile()
        self._excel_repo: ExcelFile = ExcelFile()
        self._folder_repo = Folder()
        self._pkl_repo = PklFile()

    def get_an_excel_file(
        self, excel_file_path, usecols=None, sheet_name=None, header=0
    ):
        """
        [注意]:    拡張子も含んだパスを記述すること!
        """
        return self._excel_repo.read(
            excel_file_path, sheet_name=sheet_name, use_cols=usecols, header=header
        )

    def upload_an_excel_file(self, output_path, vpp_data, excel_file_name) -> None:
        """
        拡張子を含める必要はない。
        """
        return self._excel_repo.upload(
            dict_df=vpp_data.convert_dict_df(), path=output_path, name=excel_file_name
        )

    def get_a_csv_file(self, csv_file_path, use_cols=None, header=0):
        """
        [注意]:    拡張子も含んだパスを記述すること!
        """
        return self._csv_repo.get(
            file_path=csv_file_path, usecols=use_cols, header=header
        )

    def upload_a_csv_file(self, df_csv, output_file_path, name) -> None:
        """
        拡張子を含める必要はない。
        """
        return self._csv_repo.upload(
            df=df_csv, file_path=output_file_path, file_name=name
        )

    def get_df_dict_datas(self, folder_path):
        dict_data: dict = {}
        objects = self.get_folder_objects(folder_path)
        for file in objects:
            dict_data[file.name[:-4]] = self.get_a_csv_file(csv_file_path=file)
        return dict_data

    def get_folder_objects(self, folder_path) -> list:
        return self._folder_repo.get(path=folder_path)

    def create_folder(self, dir_path, new_dir_name) -> str:
        return self._folder_repo.create(path=dir_path, name=new_dir_name)

    def get_a_pkl_file(self, csv_file_path):
        return self._pkl_repo.get(csv_file_path)

    def upload_a_pkl_file(self, csv_file_path, dic):
        return self._pkl_repo.get(csv_file_path, dic)
