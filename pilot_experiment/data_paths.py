from pathlib import Path
from icecream import ic

dir_path = Path(__file__).resolve().parents[0]
data_path = dir_path / "data"

RAW_DATA_PATH = data_path / "raw"
INTERIM_DATA_PATH = data_path / "interim"
PROCESSED_DATA_PATH = data_path / "processed"

if __name__ == "__main__":
    ic(RAW_DATA_PATH)
    ic(INTERIM_DATA_PATH)
    ic(PROCESSED_DATA_PATH)
