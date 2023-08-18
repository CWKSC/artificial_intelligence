import torch
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, tensor):
        return self.layer(tensor)