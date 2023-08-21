import torch
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(79, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tensor):
        return self.layer(tensor)