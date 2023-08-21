from pathlib import Path

import pandas as pd

from . import core


def read_csv(path: str) -> pd.DataFrame:
    path : Path = core.current_file_directory / (path + ".csv")
    return pd.read_csv(path)

def save_df_to_csv(dataframe: pd.DataFrame, path: str) -> None:
    filepath = core.current_file_directory / (path + ".csv")
    directory = filepath.parent
    directory.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(filepath, index=False)