
from torch import nn

class BridgeNet(nn.Module):

    def __init__(self, num_input_feature: int, num_output_feature: int, num_neuron: int, num_layer: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_input_feature, num_neuron),
            nn.ReLU(),
            *[nn.Linear(num_neuron, num_neuron) for _ in range(num_layer)],
            nn.ReLU(),
            nn.Linear(num_neuron, num_output_feature)
        )

    def forward(self, tensor):
        return self.layer(tensor)

