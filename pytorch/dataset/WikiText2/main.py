import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train = WikiText2(split='train')
valid = WikiText2(split='valid')
test = WikiText2(split='test')

train = list(train)
valid = list(valid)
test = list(test)
print('len(train):', len(train)) # 36718
print('len(valid):', len(valid)) # 3760
print('len(test): ', len(test))  # 4358
print()

print('train[0:4]:', train[0:4])
print()
print('valid[0:4]:', valid[0:4])
print()
print('test[0:4]:', test[0:4])
print()

tokenizer = get_tokenizer('basic_english')
token_lists = map(tokenizer, train)
token_lists = filter(lambda ele: len(ele) > 0, token_lists)
token_lists = list(token_lists)
print('token_lists[0:2]:', token_lists[0:2])
print()

vocab = build_vocab_from_iterator(token_lists, specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

print('len:', len(vocab))
print()

token_id_lists = list(map(vocab, token_lists))
token_id_tensors = list(map(torch.tensor, token_id_lists))
token_id_tensor =  torch.cat(token_id_tensors)
print(token_id_tensor.shape)
print(token_id_tensor[0:30])
print()

train_tensor = torch.cat([torch.tensor(vocab(token_list), dtype=torch.long) for token_list in token_lists])
print(train_tensor.shape)
print(train_tensor[0:30])
print()

