import torch
from torch import nn
import pandas as pd
import data_processing as dp
import artificial_neural_network as ann
from Model import Model
dp.init(__file__)
ann.init(__file__)

train_df = dp.read_csv("processed/train") 
test_df = dp.read_csv("processed/test")

target_df, input_df = dp.spilt_df(train_df, ['Transported'])
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])

target_tensors = dp.toTensors(target_df)
input_tensors = dp.toTensors(input_df)
test_tensors = dp.toTensors(test_df)

model = Model()

ann.train(
    model,
    target_tensors,
    input_tensors,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN32'
)

# predict_list = ann.predict(
#     model,
#     test_tensors
# )

# result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
# dp.save_df_to_csv(result_df, 'submission/NN')

