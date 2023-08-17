import torch
from torch import nn
import pandas as pd
import data_processing as dp
import artificial_neural_network as ann
dp.init(__file__)
ann.init(__file__)

train_df = dp.read_csv("processed/train") 
test_df = dp.read_csv("processed/test")

target_df, input_df = dp.spilt_df(train_df, ['Transported'])
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])

target_tensors = dp.toTensors(target_df)
input_tensors = dp.toTensors(input_df)
test_tensors = dp.toTensors(test_df)

def correct_func(pred, target):
    pred = pred.item()
    target = target.item()
    pred = 1 if pred > 0.5 else 0
    return pred == target

model = nn.Sequential(
    nn.Linear(13, 13),
    nn.Linear(13, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 13),
    nn.Linear(13, 1),
)

ann.train(
    target_tensors,
    input_tensors,
    model,
    correct_func=correct_func
)

predict_list = ann.predict(
    test_tensors,
    model
)

result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
dp.save_df_to_csv(result_df, 'submission/NN')

