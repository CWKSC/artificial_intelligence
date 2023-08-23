import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer

decoder_layer = TransformerDecoderLayer(d_model=32, nhead=8)
transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

# memory = torch.rand(10, 32, 512)
# tgt = torch.rand(20, 32, 512)
memory = torch.rand(32, 32)
tgt = torch.rand(32, 32)
out = transformer_decoder(tgt, memory)

print(out)
