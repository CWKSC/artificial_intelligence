import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import data_processing as dp
import artificial_neural_network as ann
from Model import Model
dp.init(__file__)
ann.init(__file__)

test_df = dp.read_csv("processed/test")
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])
test_tensors = dp.df_to_2d_tensor(test_df)

model = ann.load_model(Model(), 'model/NN32')

predict_list = ann.predict(
    model,
    test_tensors
)

result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
dp.save_df_to_csv(result_df, 'submission/NN32')