from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from tabulate import tabulate
from sigfig import round

import data_processing as dp


class DataType:
    def __str__(self) -> str:
        pass


class Numeric(DataType):
    def __str__(self) -> str:
        return 'Numeric'

class ZeroOneOnly(Numeric):
    def __str__(self):
        return 'Numeric - ZeroOne'


class Category(DataType):
    def __str__(self):
        return 'Category'
    
class TrueFalseOnly(Category):
    def __str__(self):
        return 'Category - TrueFalse'


class Mixing(DataType):
    def __str__(self):
        return 'Mixing'

class Unknown(DataType):
    def __str__(self):
        return 'Unknown'


numeric_types = [
    int,
    float,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64
]

category_types = [
    str,
    bool
]

def to_datatype(ele: Any) -> DataType:
    ele_type = type(ele)

    if ele_type in numeric_types:
        return Numeric()
    
    if ele_type in category_types:
        return Category()
    
    print('[Debug - to_datatype - Unknown]', ele_type, "|", ele, "| end")
    return Unknown


def analysis_series_type(series: pd.Series) -> DataType:
    series = series.dropna()
    first_cell_value = series.iloc[0]
    first_cell_value_str = str(first_cell_value)

    # Check Zero One only series
    if first_cell_value_str == '0' or first_cell_value_str == '0.0' or first_cell_value_str == '1' or first_cell_value_str == '1.0':
        is_all_zero_or_one = True
        for cell in series:
            cell_str = str(cell)
            if cell_str != '0' and cell_str != '0.0' and cell_str != '1' and cell_str != '1.0':
                is_all_zero_or_one = False
        if is_all_zero_or_one:
            return ZeroOneOnly()

    # Check True False only series
    if first_cell_value_str.lower() == 'true' or first_cell_value_str.lower() == 'false':
        is_all_zero_or_one = True
        for cell in series:
            cell_str = str(cell)
            if cell_str.lower() == 'true' and cell_str.lower() == 'false':
                is_all_zero_or_one = False
        if is_all_zero_or_one:
            return TrueFalseOnly()

    first_cell_type = to_datatype(first_cell_value)

    for cell in series:
        cell_type = to_datatype(cell)
        if type(cell_type) is not type(first_cell_type):
            print('[Debug - analysis_series_type - Mixing]', first_cell_type, cell_type)
            return Mixing()
    return first_cell_type

def series_to_tabulate_data_row(series: pd.Series):

    datatype = analysis_series_type(series)
    datatype_str = str(datatype)
    num_na = series.isna().sum()
    is_unique_str = str(series.is_unique)

    value_counts = series.value_counts()
    mode = value_counts.idxmax()
    num_max_repeat = value_counts[mode]
    if isinstance(datatype, Numeric):
        mode = round(mode, sigfigs=4, warn=False)
    else:
        mode = str(mode)
        if len(mode) > 15:
            mode = mode[:11] + ' ...'

    if isinstance(datatype, Numeric):
        min_value = round(series.min(), sigfigs=4, warn=False)
        max_value =  round(series.max(), sigfigs=4, warn=False)
        mean_value =  round(series.mean(), sigfigs=4, warn=False)
        std_value =  round(series.std(), sigfigs=4, warn=False)
    else:
        min_value = ''
        max_value = ''
        mean_value = ''
        std_value = ''
    
    return [datatype_str, num_na, is_unique_str, mode, num_max_repeat, min_value, max_value, mean_value, std_value]

def analysis_df(df: pd.DataFrame):
    headers = ['Column', 'Data type', 'Na', 'Unique', 'Mode', 'num', 'Min', 'Max', 'Mean', 'Std']
    tabulate_data = []
    for column in df:
        series = df[column]
        tabulate_data_row = series_to_tabulate_data_row(series)
        tabulate_data_row = [column, *tabulate_data_row]
        tabulate_data.append(tabulate_data_row)
    print(tabulate(tabulate_data, headers))
    print()


def analysis_train_test(
    data_dir_path: str = "data",
    id_field: str = 'id',
    target_field: str = 'target', 
):
    train_df = dp.read_csv(f"{data_dir_path}/train")
    test_df = dp.read_csv(f"{data_dir_path}/test")

    train_column_names = train_df.columns.tolist()
    test_column_names = test_df.columns.tolist()

    print()
    print(train_df)
    print(train_column_names)
    print()
    print(test_df)
    print(test_column_names)
    print()

    id_series = pd.concat([train_df[id_field], test_df[id_field]])
    target_series = train_df[target_field]

    data_df = pd.concat([
        train_df.drop(columns=[id_field, target_field]),
        test_df.drop(columns=[id_field])
    ], join="inner")

    col_max_len = max(map(len, data_df.columns))

    print(data_df)
    print()

    headers = ['Column', 'Data type', 'Na', 'Unique', 'Mode', 'num', 'Min', 'Max', 'Mean', 'Std']

    print(tabulate([
        [id_field, *series_to_tabulate_data_row(id_series)],
        [target_field, *series_to_tabulate_data_row(target_series)]
    ], headers=headers))
    print()

    analysis_df(data_df)

    template_str = f"""import pandas as pd
import data_analysis as da
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')

def process(df: pd.DataFrame) -> pd.DataFrame:
    dp.transform(
        df,
"""
    
    for column in data_df:
        series = data_df[column]
        datatype = analysis_series_type(series)
        isna = series.isna().any()
        is_unique = series.is_unique

        message = ""
        column = f'\'{column}\''
        message += f'        ({column: <{col_max_len + 2}}'

        if isinstance(datatype, Numeric):
            if isna:
                message += ', dp.FillNaWithMean()'
        if isinstance(datatype, Category):
            if is_unique:
                message += ', dp.Drop()'
            else:
                if isna:
                    message += ', dp.FillNaWithMode()'
                message += ', dp.VocabEncode()'
        
        message += '),\n'
        template_str += message

    template_str += f"""    )
    da.analysis_df(df)
    dp.transformAll(df, dp.Apply(float), except_columns = ['{id_field}'])
    return df

train_df = process(train_df)
test_df = process(test_df)

train_df.drop(columns=['{id_field}'], inplace=True)
dp.save_df_to_csv(train_df, '../processed/train')
train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['{target_field}'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['{id_field}'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')

"""
    print(template_str)

    
