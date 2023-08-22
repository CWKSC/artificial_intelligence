import pandas as pd

import data_processing as dp


def random_sampling(
    df: pd.DataFrame, 
    target_field: str = 'target', 
    valid_ratio: float = 0.5, 
    n: int = 30
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    result = []
    for _ in range(n):
        valid_df = df.sample(frac = valid_ratio)
        train_df = df.drop(index=valid_df.index)
        train_target_df, train_input_df = dp.spilt_df(train_df, columns=[target_field])
        valid_target_df, valid_input_df = dp.spilt_df(valid_df, columns=[target_field])
        result.append((train_input_df, train_target_df, valid_input_df, valid_target_df))
    return result


def k_fold(
    df: pd.DataFrame, 
    target_field: str = 'target', 
    n: int = 5
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    num_row = df.shape[0]
    chunk_size = num_row / n 
    result = []
    for i in range(5):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        valid_df = df.iloc[start: end]
        train_df = df.drop(index=valid_df.index)
        train_target_df, train_input_df = dp.spilt_df(train_df, columns=[target_field])
        valid_target_df, valid_input_df = dp.spilt_df(valid_df, columns=[target_field])
        result.append((train_input_df, train_target_df, valid_input_df, valid_target_df))
    return result