import os
from pathlib import Path


class BaseDataProcesser:
    def __init__(self) -> None:
        dir_path = Path(__file__).resolve().parents[1]
        self.data_path = dir_path / "data"
        self.raw_data_path = self.data_path / "raw"
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.intermin_data_path = self.data_path / "interim"
        self.intermin_data_path.mkdir(parents=True, exist_ok=True)

        self.processed_data_path = self.data_path / "processed"
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        self.raw_data_files = os.listdir(self.raw_data_path)
