import torch
from torch import nn
import pandas as pd
import data_processing as dp
dp.init(__file__)

train_dataframe = dp.read_csv("processed/train")
test_dataframe = dp.read_csv("processed/test")

train_tensors = dp.toTensors(train_dataframe)
test_tensors = dp.toTensors(test_dataframe)

# Get cpu, gpu or mps device for training.
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RAdam(model.parameters())

# RAdam
# 30 - 368
# 100 - 297

# Adam
# 30 - 374
# 100 - 304

# AdamW
# 30 - 397
# 100 - 339

# NAdam
# 30 - 401
# 100 - 340

# Adamax
# 30 - 444
# 100 - 

# Adagrad
# 30 - 426
# 100 - 341

# RMSprop
# 30 - 493

# Rprop
# 30 - 17967


def train(model, loss_fn, optimizer):
    model.train()
    loss_count = 0
    correct = 0
    for tensor in train_tensors:
        target = torch.Tensor([tensor[0]]).to(device)
        input = tensor[1:].to(device)

        pred = model(input)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred[0].item()
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        if pred == int(target[0].item()):
            correct += 1

        loss_count += loss.item()

    print(f"loss: {loss_count:>7f} ")
    print(f"correct: {correct / len(train_tensors)}  {correct} / {len(train_tensors)}")

def test(model, loss_fn):
    model.eval()
    output = []
    with torch.no_grad():
        for tensor in test_tensors:
            passengerId = str(int(tensor[0].item()))
            input = tensor[1:].to(device)
            survived = model(input)
            survived = survived[0].item()
            if survived > 0.5:
                survived = "1"
            else:
                survived = "0"
            output.append([passengerId, survived]) 
    return output

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, loss_fn, optimizer)
print("Done!")
output = test(model, loss_fn)
print(output)


df = pd.DataFrame(output, columns=['PassengerId', 'Survived'])
dp.save_df_to_csv(df, 'submission/NN')