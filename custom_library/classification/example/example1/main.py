import data_processing as dp
import classification as cla
import cross_validation as cv
dp.init(__file__)

train_df = dp.read_csv('data/train')

result = cla.try_classifier(train_df, n = 30)
print(result)
