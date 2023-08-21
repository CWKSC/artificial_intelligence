import torch
from torch import nn
import pandas as pd
import data_processing as dp
import artificial_neural_network as ann
from Model import Model
dp.init(__file__)
ann.init(__file__)

train_df = dp.read_csv('processed/train')
target_df, input_df = dp.spilt_df(train_df, ['Transported'])
target_tensors = dp.toTensors(target_df)
input_tensors = dp.toTensors(input_df)

model = ann.load_model(Model(), 'model/NN32')

predict_list = ann.predict(model, input_tensors)
predict_list = [1 if ele.item() > 0.5 else 0 for ele in predict_list]
correct = [1 if predict_list[i] == target_tensors[i].item() else 0 for i in range(0, len(predict_list))]

model_check = Model()

correct_df = train_df.copy()
correct_df['Correct'] = correct
# print(correct_df)
_, target_df, input_df = dp.spiltN_df(correct_df, [['Transported'], ['Correct']])
target_tensors = dp.toTensors(target_df)
input_tensors = dp.toTensors(input_df)

ann.train(
    model_check,
    target_tensors,
    input_tensors,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN32_check'
)




not_correct_only_df = correct_df[correct_df['Correct'] == 0].copy()
# print(not_correct_only_df)
target_df, _, input_df = dp.spiltN_df(not_correct_only_df, [['Transported'], ['Correct']])
target_tensors = dp.toTensors(target_df)
input_tensors = dp.toTensors(input_df)

model2 = Model()

ann.train(
    model2,
    target_tensors,
    input_tensors,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN32_2'
)


