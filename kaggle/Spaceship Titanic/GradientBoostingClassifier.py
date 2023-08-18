from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import data_processing as dp
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv("processed/train")
test_df = dp.read_csv("processed/test")

target_df, input_df = dp.spilt_df(train_df, ['Transported'])
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])

inputs = input_df
targets =  target_df['Transported'].tolist()
tests = test_df

model = GradientBoostingClassifier(
    n_estimators=100
)
model.fit(inputs, targets)
print(model.score(inputs, targets))

predictions = model.predict(tests)
print(predictions)
predictions = ['True' if pred else 'False' for pred in predictions]

result_df['Transported'] = predictions
dp.save_df_to_csv(result_df, "submission/GradientBoostingClassifier")
