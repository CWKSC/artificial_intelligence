
from torch import Tensor, exp, arange, zeros, sin, cos
from torch.nn import Dropout, Module, Embedding
from math import log, sqrt

# From: Language Translation with nn.Transformer and torchtext

class EmbeddingV1(Module):
    def __init__(self, vocab_size: int, emb_size):
        super(EmbeddingV1, self).__init__()
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * sqrt(self.emb_size)