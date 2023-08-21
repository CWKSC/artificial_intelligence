
from pathlib import Path
from typing import Any
import traceback

import numpy
import pandas as pd
import torch

from .operation import Drop, Operation

def transform(dataframe: pd.DataFrame, *process_chains: tuple[tuple[str, list[Operation]]]):
    for process_chain in process_chains:
        if type(process_chain) != tuple:
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
