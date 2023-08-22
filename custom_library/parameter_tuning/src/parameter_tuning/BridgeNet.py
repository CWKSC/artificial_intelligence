from typing import Callable
from artificial_neural_network import BridgeNet

basic_pbounds_BridgeNet: dict[str, tuple[float, float]] = {
    'num_neuron': (2, 512),
    'num_layer': (1, 8)
}

def basic_model_setter_BridgeNet_creator(num_input_feature: int, num_output_feature: int) -> Callable:
    def basic_model_setter_BridgeNet(**args):
        return BridgeNet(
            num_input_feature = num_input_feature,
            num_output_feature = num_output_feature,
            num_neuron = int(args['num_neuron']),
            num_layer = int(args['num_layer']),
        )
    return basic_model_setter_BridgeNet
