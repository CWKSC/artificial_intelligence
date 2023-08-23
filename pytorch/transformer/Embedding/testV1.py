
from EmbeddingV1 import EmbeddingV1
import torch

input = torch.rand(8)
print(input)

embedding = EmbeddingV1(16, 4)
output = embedding(input)
print(output)

