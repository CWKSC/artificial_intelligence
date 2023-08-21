from catboost import CatBoostRegressor
import data_processing as dp
import numpy as np
import pandas as pd
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 1024 0.001 -> 2.95972
# 1024 0.002 -> 2.63696
# 1024 0.008 -> 1.65781
# 1024 0.016 -> 1.54868
# 2048 0.016 -> 1.59305
# 2048 0.008 -> 1.55381
# 1024 0.012 -> 1.54966
# 512 0.012 -> 1.74035
# 2048 0.012 -> 1.58512
# 

model = CatBoostRegressor(
    iterations=2048,
    learning_rate=0.012,
    verbose=10
)
model.fit(input_df, target_df.to_numpy().ravel())
# print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['sales'] = predictions
print(test_target_df['sales'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostRegressor")