from pathlib import Path
from typing import Any

import numpy
import pandas as pd
import torch

current_file_directory: Path = None


def init(file_path: str) -> None:
    global current_file_directory
    current_file_directory = Path(file_path).parent


def spilt_df(dataframe: pd.DataFrame, columns: list[str] = []) -> tuple[ pd.DataFrame, pd.DataFrame]:
    df1 = dataframe[columns]
    df2 = dataframe.drop(columns=columns)
    return df1, df2


def spiltN_df(dataframe: pd.DataFrame, columns_list: list[list[str]]=[]) -> list[pd.DataFrame]:
    df_list = []
    temp_df = dataframe.copy()
    for columns in columns_list:
        df_list.append(dataframe[columns])
        temp_df = temp_df.drop(columns=columns)
    df_list.append(temp_df)
    return df_list