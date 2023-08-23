from torch import zeros, arange, float, exp, sin, cos
from torch.nn import Module
from math import log

# From: Build your own Transformer from scratch using Pytorch

class PositionalEncodingV3(Module):

    def __init__(self, 
            d_model,
            max_seq_length
        ):
        super(PositionalEncodingV3, self).__init__()
        
        pe = zeros(max_seq_length, d_model)
        position = arange(0, max_seq_length, dtype=float).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2).float() * -(log(10000.0) / d_model))
        
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]