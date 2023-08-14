from pathlib import Path
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
        print('Vocab:', ', '.join(vocab.get_itos()))
        encoded = vocab(series_list)
        return pd.Series(encoded)

class Map(Operation):
    def __init__(self, mappingTable: dict[str, str]) -> None:
        super().__init__()
        self.mappingTable = mappingTable
    def name(self) -> str:
        return 'Map'
    def operate(self, series: pd.Series) -> pd.Series:
        series_list = series.apply(str).tolist()
        print(', '.join([f'{key} -> {value}' for key, value in self.mappingTable.items()]))
        return pd.Series(series_list).map(self.mappingTable)

class Apply(Operation):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func
    def name(self) -> str:
        return 'Apply'
    def operate(self, series: pd.Series) -> pd.Series:
        series_list = series.apply(str).tolist()
        return pd.Series(series_list).apply(self.func)

class FillNa(Operation):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value
    def name(self) -> str:
        return 'FillNa'
    def operate(self, series: pd.Series) -> pd.Series:
        return series.fillna(self.value)

class Drop(Operation):
    def name(self) -> str:
        return 'Drop'

def init(file_path: str):
    global current_file_directory
    current_file_directory = Path(file_path).parent

def analysis(dataframe: pd.DataFrame):
    print()
    print("Columns:")
    column_names = dataframe.columns.tolist()
    print(column_names)
    print("Number of row:", dataframe.count().max())
    print()

    padding = max(map(len, column_names)) + 4

    # Unique
    for column_name in dataframe:
        series = dataframe[column_name]
        series_na = series.isna()
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
        series_na = series.isna()
        if series_na.any():
            print(f'{column_name: <{padding}} have {series_na.sum()} missing value')
    print()

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(current_file_directory / (path + ".csv"))

def save_csv(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_csv(current_file_directory / (path + ".csv"), index=False)

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
            series = operation.operate(series)
        
        if not is_drop:
            dataframe[column_name] = series
        
        print()


def transformAll(dataframe: pd.DataFrame, *operations: tuple[Operation]):
    for column_name in dataframe:
        series = dataframe[column_name]
        for operation in operations:
            series = operation.operate(series)
        dataframe[column_name] = series

def toTensors(dataframe: pd.DataFrame) -> list[torch.Tensor]:
    return torch.Tensor(dataframe.to_numpy())
