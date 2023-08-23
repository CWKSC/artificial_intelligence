import torch
from PositionalEncodingV3 import PositionalEncodingV3

embedding_vector_len = 4
max_seq_length = 5000
input = torch.zeros(embedding_vector_len).unsqueeze(0)

positional_encoding = PositionalEncodingV3(embedding_vector_len, max_seq_length)
output = positional_encoding(input)
print(output)


