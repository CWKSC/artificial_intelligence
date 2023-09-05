import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import torch
import numpy as np 
from torch import nn

from artificial_neural_network import BridgeNet
import artificial_neural_network as ann
import data_processing as dp
ann.init(__file__)
dp.init(__file__)

input_df = dp.read_csv("../processed/train_input")
target_df = dp.read_csv("../processed/train_target")
test_input_df = dp.read_csv("../processed/test_input")
test_target_df = dp.read_csv("../processed/test_target")

nn_inputs = dp.df_to_2d_tensor(input_df)
nn_targets = dp.df_to_2d_list(target_df)
# test_inputs = dp.df_to_2d_tensor(test_input_df).to(ann.device)

nn_targets = [[0.0, 1.0] if target[0] < 0.5 else [1.0, 0.0] for target in nn_targets]
nn_targets = torch.tensor(nn_targets)
# print(targets)
# print(torch.tensor(targets))

# model = BridgeNet(10, 2, 6, 1)
model =  nn.Sequential(
    nn.Linear(10, 10),
    *[nn.Linear(10, 10) for _ in range(10)],
    nn.Linear(10, 2)
)

repeat = 30
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RAdam(model.parameters())

def correct_func(prediction, target):
    prediction = [0 if pred < 0.5 else 1 for pred in prediction]
    target = [0 if tar < 0.5 else 1 for tar in target]
    #  print(prediction)
    # print(target)
    # print(prediction == target)
    return prediction == target

ann.train(
    model,
    nn_inputs,
    nn_targets,
    repeat = 10000,
    correct_func=correct_func,
    save_file_name='NN'
)

nn_predictions = ann.predict(
    model,
    nn_inputs
)

svm_inputs = nn_predictions
svm_targets = target_df.to_numpy().ravel()

print(svm_inputs)
print(svm_targets)

clf = svm.SVC()
clf.fit(svm_inputs, svm_targets)
print(clf.score(svm_inputs, svm_targets))

nn_test_input = dp.df_to_2d_tensor(test_input_df)
nn_test_predictions = ann.predict(
    model,
    nn_test_input
)
svm_predictions = clf.predict(nn_test_predictions)

test_target_df['Survived'] = svm_predictions.astype(int)
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, "../submission/NN2SVM")
