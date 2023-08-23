import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

print(generate_square_subsequent_mask(1))
print(generate_square_subsequent_mask(2))
print(generate_square_subsequent_mask(3))

embedding_vector_len = 512
src_emb = torch.zeros(embedding_vector_len)
tgt_emb = torch.zeros(embedding_vector_len)

print(create_mask(src_emb, tgt_emb))

print(torch.nn.Transformer.generate_square_subsequent_mask(1))
print(torch.nn.Transformer.generate_square_subsequent_mask(2))
print(torch.nn.Transformer.generate_square_subsequent_mask(3))

