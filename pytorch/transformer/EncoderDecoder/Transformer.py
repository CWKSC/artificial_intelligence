import torch
from torch import nn, Tensor
from typing import Any, Callable, Union
from torch.nn.functional import F
import math

class PositionalEncoding(nn.Module):

    def __init__(self,
            embedding_vector_len: int,
            dropout_p: float = 0.1,
            maxlen: int = 5000
        ):
        super().__init__()

        den = torch.exp(-torch.arange(0, embedding_vector_len, 2) * math.log(10000) / embedding_vector_len)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_vector_len))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    def __init__(self,
            source_vocab_len: int,
            target_vocab_len: int,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            embedding_vector_len: int = 512,
            maxlen: int = 5000,
            num_head: int = 8,
            dropout_p: float = 0.1,
            dim_feedforward: int = 2048,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 0.00001,
            batch_first: bool = False,
            norm_first: bool = False,
            device: Any | None = None,
            dtype: Any | None = None,
            norm: Any | None = None
        ):
        super().__init__()

        self.source_embedding = nn.Embedding(
            num_embeddings = source_vocab_len, 
            embedding_dim = embedding_vector_len
        )
        self.target_embedding = nn.Embedding(
            num_embeddings = target_vocab_len, 
            embedding_dim = embedding_vector_len
        )
        self.positional_encoding = PositionalEncoding(
            embedding_vector_len = embedding_vector_len,
            dropout_p = dropout_p,
            maxlen = maxlen
        )

        self.transformer = nn.Transformer(
            d_model = embedding_vector_len,
            nhead = num_head,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout_p
        )

        self.linear = nn.Linear(embedding_vector_len, target_vocab_len)
        self.softmax = nn.Softmax()

    def forward(self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            memory_key_padding_mask: Tensor
        ):
        source_embedding_vector = self.positional_encoding(self.source_embedding(src))
        target_embedding_vector = self.positional_encoding(self.target_embedding(trg))
        mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        outs = self.transformer(source_embedding_vector, target_embedding_vector, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.linear(outs)
    