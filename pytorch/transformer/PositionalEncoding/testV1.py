import torch
from PositionalEncodingV1 import PositionalEncodingV1

embedding_vector_len = 4
dropout_p = 0.1
input = torch.zeros(embedding_vector_len)

positional_encoding = PositionalEncodingV1(embedding_vector_len, dropout_p)
output = positional_encoding(input)
print(output)


