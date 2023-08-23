import torch

def get_key_padding_mask(tensor, padding_id):
    key_padding_mask = torch.zeros(tensor.shape)
    key_padding_mask[tensor == padding_id ] = -torch.inf
    return key_padding_mask

tensor = torch.Tensor([0, 1, 3, 4, 2, 2, 2, 2])

print(get_key_padding_mask(tensor, 2))

