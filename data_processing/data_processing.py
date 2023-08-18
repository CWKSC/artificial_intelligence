from decimal import Decimal, InvalidOperation
from pathlib import Path
import traceback
from typing import Any
import numpy
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import torch

current_file_directory: Path = None

class Operation:
    def name(self) -> str:
        return ""
    def operate(self, series: pd.Series) -> pd.Series:
        pass

class VocabEncode(Operation):
    def name(self) -> str:
        return "VocabEncode"
    def operate(self, series: pd.Series) -> pd.Series:
        series_list = series.apply(str).tolist()
        vocab = build_vocab_from_iterator([series_list])
        itos = vocab.get_itos()
        if len(itos) > 10:
            print('Vocab:', ', '.join(itos[:10]))
        else:
            print('Vocab:', ', '.join(itos))
        encoded = vocab(series_list)
        print(encoded[:20])
        output = pd.Series(encoded)
        # print(output)
        return output

class Replace(Operation):
    def __init__(self, mappingTable: dict[str, Any]) -> None:
        super().__init__()
        self.mappingTable = mappingTable
    def name(self) -> str:
        return 'Replace'
    def operate(self, series: pd.Series) -> pd.Series:
        series_str = series.apply(str)
        print(', '.join([f'{key} -> {value}' for key, value in self.mappingTable.items()]))
        for key, value in self.mappingTable.items():
            series_str.replace(key, value, inplace=True)
        return series_str

class Apply(Operation):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func
    def name(self) -> str:
        return 'Apply'
    def operate(self, series: pd.Series) -> pd.Series:
        return series.apply(str).apply(self.func)

class FillNa(Operation):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value
    def name(self) -> str:
        return 'FillNa'
    def operate(self, series: pd.Series) -> pd.Series:
        return series.fillna(self.value)

class FillNaWithMean(Operation):
    def __init__(self) -> None:
        super().__init__()
    def name(self) -> str:
        return 'FillNaWithMean'
    def operate(self, series: pd.Series) -> pd.Series:
        mean = series.dropna().mean()
        print(f'Mode: {mean}')
        return series.fillna(mean)

class FillNaWithMode(Operation):
    def __init__(self) -> None:
        super().__init__()
    def name(self) -> str:
        return 'FillNaWithMode'
    def operate(self, series: pd.Series) -> pd.Series:
        mode = series.dropna().mode()
        print(f'Mode: {mode}')
        return series.fillna(mode)


class Drop(Operation):
    def name(self) -> str:
        return 'Drop'

def init(file_path: str):
    global current_file_directory
    current_file_directory = Path(file_path).parent

def analysis_gui(dataframe: pd.DataFrame):
    pass

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
            display_str += ', dp.Map({\'True\': 1, \'False\': 0})'
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
    print('dp.save_df_to_csv(temp_df, \'processed/temp\')')
    print()
    


def read_csv(path: str) -> pd.DataFrame:
    path : Path = current_file_directory / (path + ".csv")
    return pd.read_csv(path)

def save_df_to_csv(dataframe: pd.DataFrame, path: str) -> None:
    filepath = current_file_directory / (path + ".csv")
    directory = filepath.parent
    directory.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(filepath, index=False)

def transform(dataframe: pd.DataFrame, *process_chains: tuple[tuple[str, list[Operation]]]):
    for process_chain in process_chains:
        if type(process_chain) != tuple:
            print("[Warning] process_chain not a tuple")
            continue

        column_name, *operations = process_chain
        operation_names = list(map(lambda ele: ele.name(), operations))

        is_drop = False
        series = dataframe[column_name]
        for i, operation in enumerate(operations):
            operation_names[i] = f'[{operation_names[i]}]'
            print(" -> ".join([column_name, *operation_names]))
            operation_names[i] = operation.name()

            if type(operation) is Drop:
                is_drop = True
                dataframe.drop(columns=[column_name], inplace=True)
                break
            series = operation.operate(series)
            # print(series)

        if not is_drop:
            dataframe[column_name] = series.tolist()
            # print(dataframe)
        
        print()


def transformAll(dataframe: pd.DataFrame, *operations: tuple[Operation], except_columns: list[str] = []):
    for column_name in dataframe:
        if column_name in except_columns:
            continue
        try:
            series = dataframe[column_name]
            for operation in operations:
                series = operation.operate(series)
            dataframe[column_name] = series
        except Exception as e:
            traceback.print_exc()
            print("Error in column: ", column_name)

def toTensors(dataframe: pd.DataFrame) -> torch.Tensor:
    return torch.Tensor(dataframe.to_numpy())


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