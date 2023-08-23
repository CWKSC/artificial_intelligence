from torch import Tensor, exp, arange, zeros, sin, cos
from torch.nn import Dropout, Module
from math import log

# From: Language Translation with nn.Transformer and torchtext

class PositionalEncodingV1(Module):

    def __init__(self,
            embedding_vector_len: int,
            dropout_p: float,
            maxlen: int = 5000
        ):
        super(PositionalEncodingV1, self).__init__()

        den = exp(-arange(0, embedding_vector_len, 2) * log(10000) / embedding_vector_len)
        pos = arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = zeros((maxlen, embedding_vector_len))
        pos_embedding[:, 0::2] = sin(pos * den)
        pos_embedding[:, 1::2] = cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = Dropout(dropout_p)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

