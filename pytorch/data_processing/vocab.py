import torchtext

unk = '<unk>'

sentences = [['a', "b", 'c', 'd', '<unk>']] # Iterable[Iterable[str]]
vocab = torchtext.vocab.build_vocab_from_iterator(sentences, min_freq=1, max_tokens=30, specials=[unk])
vocab.set_default_index(vocab[unk])

print('len:', len(vocab))
print(vocab.get_stoi())
print(vocab.get_itos())
print("vocab['c']:", vocab['c'])
print("vocab['www']:", vocab['www'])
print()


from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer

train = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
token_lists = list(map(tokenizer, train))

vocab = torchtext.vocab.build_vocab_from_iterator(token_lists, min_freq=1, specials=[unk])
vocab.set_default_index(vocab[unk])

print('len:', len(vocab))
# print(vocab.get_stoi())
# print(vocab.get_itos())
print("vocab['cat']:", vocab['cat'])
print("vocab['ahsdkjashdkjasd']:", vocab['ahsdkjashdkjasd'])
print("vocab([]):", vocab([]))
print("vocab(['cat', 'played', 'ball']):", vocab(['cat', 'dog', 'a']))


