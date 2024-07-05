""" ペリーで実行する際のファイル """

import pandas as pd

from schemas import FolderPathModel

folder_model = FolderPathModel
test_df = pd.read_csv(folder_model.base_data_path / "train.csv")
print(test_df.shape)
