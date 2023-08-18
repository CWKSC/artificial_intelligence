import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import data_processing as dp
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv("processed/train")
test_df = dp.read_csv("processed/test")

target_df, input_df = dp.spilt_df(train_df, ['Transported'])
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])

train_data = lgb.Dataset(input_df, label=target_df)

param = {'num_leaves': 31, 'objective': 'binary'}
num_round = 1000
model = lgb.train(param, train_data, num_round)


predictions = model.predict(test_df)
predictions = ['True' if pred > 0.5 else 'False' for pred in predictions]

result_df['Transported'] = predictions
dp.save_df_to_csv(result_df, "submission/lgb")


