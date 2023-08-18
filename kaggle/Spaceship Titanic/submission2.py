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
test_tensors = dp.toTensors(test_df)

model = ann.load_model(Model(), 'model/NN32')
model_check = ann.load_model(Model(), 'model/NN32_check')
model2 = ann.load_model(Model(), 'model/NN32_2')


predict_list = []
with torch.no_grad():
    for input in tqdm(test_tensors):
        input = input.to(ann.device)

        is_correct = model_check(input)
        if is_correct.item() > 0.5:
            predict = model(input)
        else:
            predict = model2(input)
        
        predict_list.append(predict)

result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
dp.save_df_to_csv(result_df, 'submission/NN32_2')