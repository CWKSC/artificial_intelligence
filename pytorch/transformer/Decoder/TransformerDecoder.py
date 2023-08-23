from typing import Any, Callable, Union
import math
import torch
from torch import nn, Tensor
from torch.nn.functional import F

class PositionalEncoding(nn.Module):

    def __init__(self,
            embedding_vector_len: int,
            dropout_p: float,
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


class TransformerDecoder(nn.Module):

    def __init__(self,
            input_vocab_len: int,
            ouput_vocab_len: int,
            embedding_vector_len: int = 512,
            maxlen: int = 5000,
            num_head: int = 8,
            num_layer: int = 6,
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
        
        self.embedding = nn.Embedding(
            num_embeddings = input_vocab_len, 
            embedding_dim = embedding_vector_len
        )
        self.positionalEncoding = PositionalEncoding(
            embedding_vector_len = embedding_vector_len,
            dropout_p = dropout_p,
            maxlen = maxlen
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = embedding_vector_len, 
            nhead = num_head,
            dropout = dropout_p,
            dim_feedforward = dim_feedforward,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            norm_first = norm_first,
            device = device,
            dtype = dtype
        )
        self.decoder  = nn.TransformerDecoder(
            decoder_layer = decoder_layer, 
            num_layers = num_layer,
            norm = norm
        )

        self.linear = nn.Linear(embedding_vector_len, ouput_vocab_len)
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
        pass

