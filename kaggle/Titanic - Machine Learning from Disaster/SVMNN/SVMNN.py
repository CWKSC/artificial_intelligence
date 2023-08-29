import numpy
from sklearn import svm
import torch
import data_processing as dp
import artificial_neural_network as ann
dp.init(__file__)
ann.init(__file__)

from torch import nn
from tqdm import tqdm

class RectangleNet(nn.Module):

    def __init__(self, num_input_feature: int, num_layer: int = 1):
        super().__init__()
        self.layer = nn.Sequential(
            *[nn.Linear(num_input_feature, num_input_feature) for _ in range(num_layer)]
        )

    def forward(self, tensor):
        return self.layer(tensor)


input_df = dp.read_csv("../processed/train_input")
target_df = dp.read_csv("../processed/train_target")
test_inputs = dp.read_csv("../processed/test_input")
test_target_df = dp.read_csv("../processed/test_target")

inputs = dp.df_to_2d_tensor(input_df)
targets = dp.df_to_2d_tensor(target_df)

model = RectangleNet(10, 1)

model.to(ann.device)
model.train()

best_loss = float('inf')
best_accuracy = 0
tensor_len = len(targets)

repeat = 30
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RAdam(model.parameters())
for epoch in range(1, repeat + 1):
    print(f'Epoch {epoch}')

    total_loss = 0
    total_correct = 0

    predictions = []
    for target, input in tqdm(list(zip(targets, inputs))):

        target = target.to(ann.device)
        input = input.to(ann.device)

        pred = model(input)
        predictions.append(pred)

    with torch.no_grad():
        svm_model = svm.SVC()
        svm_model.fit(predictions, targets.ravel())
        svm_predictions = svm_model.predict(test_inputs)

    for i in range(len(svm_predictions)):
        # print(i, len(svm_predictions))
        pred = svm_predictions[i]
        target = targets[i]

        # print(torch.tensor([pred]), target)

        loss = loss_fn(torch.tensor([pred], requires_grad=True), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not numpy.isnan(loss.item()):
            total_loss += loss.item()
        
        pred = pred.item()
        target = target.item()
        pred = 1 if pred > 0.5 else 0
        if pred == target:
            total_correct += 1
    
    accuracy = total_correct / tensor_len

    print(f"loss:     {total_loss:>7f}, best_loss: {best_loss:>7f}")
    print(f"accuracy: {accuracy:>7f}, {total_correct} / {tensor_len}, best_accuracy: {best_accuracy:>7f}")

