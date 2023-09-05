from pathlib import Path
from typing import Any
import traceback

import numpy
import pandas as pd
import torch


def df_to_2d_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(dataframe.to_numpy(), dtype=torch.float32)

def df_to_2d_list(df: pd.DataFrame) -> list:
    return [df.iloc[i].tolist() for i in range(df.shape[0])]


