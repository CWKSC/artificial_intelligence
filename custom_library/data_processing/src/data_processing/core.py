from pathlib import Path
from typing import Any
import traceback

import numpy
import pandas as pd
import torch

current_file_directory: Path = None


def init(file_path: str):
    global current_file_directory
    current_file_directory = Path(file_path).parent


def analysis(dataframe: pd.DataFrame):
    print()
    print(f"{ dataframe.shape[1] } Columns, {dataframe.shape[0]} Row")
    column_names = dataframe.columns.tolist()
    print(column_names)
    print()
    print(dataframe.head(10))
    print()

    padding = max(map(len, column_names)) + 4

    # Unique
    for column_name in dataframe:
        series = dataframe[column_name]
        if series.is_unique:
            print(f'{column_name: <{padding}} is unique')
        elif series[series != pd.NA].is_unique:
            print(f'{column_name: <{padding}} is unique if ignore na value')
        else:
            value_counts = series.value_counts()
            idmax = value_counts.idxmax()
            print(f'{column_name: <{padding}} is not unique, "{idmax}" repeated {value_counts[idmax]} times')
    print()

    # Missing value
    for column_name in dataframe:
        series = dataframe[column_name]
        series_isna = series.isna()
        if series_isna.any():
            series_na = series_isna[series_isna == True]
            print(f'{column_name: <{padding}} have {series_isna.sum()} missing value, first one in {series_na.first_valid_index()}')
    print()

    print('Template:')
    print()
    print('import data_processing as dp')
    print('dp.init(__file__)')
    print()
    print('temp_df = dp.read_csv(\'data/temp\')')
    print('')
    print('dp.transform(')
    print('    temp_df,')
    for column_name in dataframe:
        series = dataframe[column_name]

        display_str = ""
        column_name_quote = f'\'{column_name}\''
        display_str += f'    ({column_name_quote: <{padding}}'

        series_dropna = series.dropna()
        cell = series_dropna[series_dropna.first_valid_index()]
        cell_type = type(cell)
        # print(cell_type)

        is_bool = cell_type == bool or cell_type == numpy.bool_
        is_float = cell_type == numpy.float64
        is_str = cell_type == str
        is_unique = series.is_unique
        have_na = series.isna().any()

        if have_na:
            if is_float or is_bool:
                display_str += ', dp.FillNa(-1)'
        if is_bool:
            display_str += ', dp.Replace({\'True\': 1, \'False\': 0})'
        if (not is_unique) and is_str:
            display_str += ', dp.VocabEncode()'
        
        display_str += '),'
        print(display_str)
    print(')')
    print('dp.transformAll(temp_df, dp.Apply(float), except_columns = [])')
    print()
    print('temp_target_df, temp_input_df = dp.spilt_df(temp_df, columns = [\'Target\'])')
    print('dp.save_df_to_csv(temp_target_df, \'processed/temp_target\')')
    print('dp.save_df_to_csv(temp_input_df, \'processed/temp_input\')')
    print()
    print('dp.analysis(temp_df)')
    print('print(temp_target_df.head(10))')
    print('print(temp_input_df.head(10))')
    print()
    


def df_to_2d_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(dataframe.to_numpy(), dtype=torch.float32)

def df_to_2d_list(df: pd.DataFrame) -> list:
    return [df.iloc[i].tolist() for i in range(df.shape[0])]

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