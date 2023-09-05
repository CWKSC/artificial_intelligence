from pathlib import Path

import pandas as pd
import numpy as np
import torch

from . import core
from . import convert

def read_csv(path: str) -> pd.DataFrame:
    path : Path = core.current_file_directory / (path + ".csv")
    return pd.read_csv(path)

def read_csv_to_ndarray(path: str) -> np.ndarray:
    df = read_csv(path)
    return df.to_numpy()

def read_csv_to_2d_tensor(path: str) -> torch.Tensor:
    df = read_csv(path)
    return convert.df_to_2d_tensor(df)

def save_df_to_csv(dataframe: pd.DataFrame, path: str) -> None:
    filepath = core.current_file_directory / (path + ".csv")
    directory = filepath.parent
    directory.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(filepath, index=False)