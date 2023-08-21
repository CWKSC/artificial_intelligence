from typing import Any

import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator


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
        display_str = ', '.join(itos[:10])
        if len(display_str) > 100:
            print('Vocab:', display_str[:100], '...')
        else:
            print('Vocab:', display_str)
        encoded = vocab(series_list)
        output = pd.Series(encoded)
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

class Drop(Operation):
    def name(self) -> str:
        return 'Drop'



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
        print(f'Mean: {mean}')
        return series.fillna(mean)


class FillNaWithMode(Operation):
    def __init__(self) -> None:
        super().__init__()
    def name(self) -> str:
        return 'FillNaWithMode'
    def operate(self, series: pd.Series) -> pd.Series:
        mode = series.dropna().mode().iloc[0]
        print(f'Mode: {mode}')
        return series.fillna(mode)

class Standardize(Operation):
    def __init__(self) -> None:
        super().__init__()
    def name(self) -> str:
        return 'Standardize'
    def operate(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        print(f'Mean: {mean}, Std: {std}')
        return (series - series.mean()) / series.std()
