from xgboost import XGBClassifier
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

xgboost_model = XGBClassifier(n_estimators=100, learning_rate= 0.3)
xgboost_model.fit(input_df, target_df)
print(xgboost_model.score(input_df, target_df))

predictions = xgboost_model.predict(test_df)
predictions = ['True' if pred > 0.5 else 'False' for pred in predictions]

result_df['Transported'] = predictions
dp.save_df_to_csv(result_df, "submission/XGBClassifier")

