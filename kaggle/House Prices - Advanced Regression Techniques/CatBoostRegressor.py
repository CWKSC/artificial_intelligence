from catboost import CatBoostRegressor
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.01,
    verbose=1000
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