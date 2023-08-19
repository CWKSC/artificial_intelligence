from catboost import CatBoostRegressor
import data_processing as dp
import numpy as np
import pandas as pd
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

model = CatBoostRegressor(
    learning_rate=0.01,
    verbose=10
)
model.fit(input_df, target_df.to_numpy().ravel())
# print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['sales'] = predictions
print(test_target_df['sales'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostRegressor")