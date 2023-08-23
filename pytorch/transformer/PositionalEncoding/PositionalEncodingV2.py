from torch import Tensor, arange, exp, zeros, sin, cos
from torch.nn import Dropout, Module
from math import log

# From: Language Modeling with nn.Transformer and torchtext

class PositionalEncodingV2(Module):

    def __init__(self, 
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000
        ):
        super().__init__()

        self.dropout = Dropout(p=dropout)

        position = arange(max_len).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = sin(position * div_term)
        pe[:, 0, 1::2] = cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
