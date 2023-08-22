import data_analysis as da
import data_processing as dp
import cross_validation as cv
dp.init(__file__)

train_df = dp.read_csv('data/train')

random_sample_list = cv.random_sampling(train_df, 'Survived')
for train_target, train_input, valid_target, valid_input in random_sample_list:
    print(train_target.shape, train_input.shape, valid_target.shape, valid_input.shape)

k_fold_sample_list = cv.k_fold(train_df, 'Survived')
for train_target, train_input, valid_target, valid_input in k_fold_sample_list:
    print(train_target.shape, train_input.shape, valid_target.shape, valid_input.shape)
