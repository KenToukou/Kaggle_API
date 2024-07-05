from pathlib import Path

from pydantic import BaseModel


class FolderPathModel(BaseModel):
    home_data_dir: Path = Path(".") / "data"
    original_data_path: Path = Path(".") / "data" / "original"
    test_data_path: Path = Path(".") / "data" / "test"
    base_data_path: Path = Path(".") / "data" / "base_data"
    output_path: Path = Path(".") / "out"
    feedback_data_home_dir: Path = Path(".") / "feedback"
