from torchtext.datasets import multi30k, Multi30k

multi30k.URL = {
    "train": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}

multi30k.MD5 = {
    "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
    "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    "test": "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36",
}

language_pair = ('en', 'de')

train = Multi30k(split="train", language_pair=language_pair)
valid = Multi30k(split="valid", language_pair=language_pair)
# test = Multi30k(split="test", language_pair=language_pair)

train = list(train)
valid = list(valid)
# test = list(test)
print(len(train)) # 29001
print(len(valid)) # 1015
# print(len(test)) #  
print()

train0 = train[0]
valid0 = valid[0]
# test0 = test[0]
print(train0)
print(valid0)
# print(test0)

train_last = train[-1]
valid_last = valid[-1]
# test_last = test[-1]
print(train_last)
print(valid_last)
# print(test_last)
