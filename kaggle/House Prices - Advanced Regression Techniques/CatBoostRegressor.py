from catboost import CatBoostRegressor

import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 1024 0.001 -> 0.26359
# 2048 0.001 -> 0.20335
# 2048 0.002 -> 0.16147
# 4096 0.002 -> 0.13996
# 4096 0.004 -> 0.13003
# 4096 0.008 -> 0.12657
# 4096 0.016 -> 0.12673
# 8192 0.008 -> 0.12592
# 16384 0.008 -> 0.12587
# 16384 0.016 -> 0.12666
# 8192 0.016 -> 0.12671
# 32768 0.008 -> 0.12581
# 65536 0.008 -> 0.1258
# 65536 0.004 -> 0.12671
# 65536 0.008 8 3 3 0.7 -> 0.1301

# 32768 0.008 -> 0.13117
# 4096 0.008 -> 0.13168
# 4096 0.0001 -> 0.34045
# 65536 0.0001 -> 0.14762
# 65536 0.0002 -> 0.13669
# 4096 0.1 -> 0.13214
# 64 0.05 -> 0.17863
# 2048 0.05 -> 0.13388

model = CatBoostRegressor(
    iterations = 2048,
    learning_rate = 0.05,
    verbose=100
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))
mean = model.feature_importances_.mean()
print('mean:', mean)
importances_feature = list(sorted(zip(model.feature_names_, model.feature_importances_), key = lambda x: x[1], reverse=True))
print(importances_feature)
print()
importances_mean = list(filter(lambda ele: ele[1] < mean, importances_feature))
print(len(importances_mean))
print([ele[0] for ele in importances_mean])

predictions = model.predict(test_input_df)

test_target_df['SalePrice'] = predictions
# print(test_target_df['SalePrice'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostRegressor")