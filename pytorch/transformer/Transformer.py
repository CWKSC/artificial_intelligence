import torch
from torch.nn import Transformer

embedding_vector_len = 4
head_num = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048

dropout_p = 0.1
input = torch.zeros(embedding_vector_len)

transformer = Transformer(
    d_model=embedding_vector_len,
    nhead=head_num,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout_p
)

src_emb = torch.zeros(embedding_vector_len)
tgt_emb = torch.zeros(embedding_vector_len)

output = transformer(
    src_emb, 
    tgt_emb, 
    src_mask, 
    tgt_mask, 
    None,
    src_padding_mask, 
    tgt_padding_mask, 
    memory_key_padding_mask
)

