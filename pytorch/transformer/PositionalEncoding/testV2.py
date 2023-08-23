import torch
from PositionalEncodingV2 import PositionalEncodingV2

embedding_vector_len = 4
dropout_p = 0.1
input = torch.zeros(3, 1, embedding_vector_len)

positional_encoding = PositionalEncodingV2(embedding_vector_len, dropout_p)
output = positional_encoding(input)
print(output)


