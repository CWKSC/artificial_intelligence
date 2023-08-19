from catboost import CatBoostRegressor
import data_processing as dp
dp.init(__file__)

train_input_df = dp.read_csv("processed/train_input")
target_df = dp.read_csv("processed/train_target")

model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.01,
    verbose=1000
)
model.fit(train_input_df, target_df.to_numpy().ravel())
print(model.score(train_input_df, target_df))
mean = model.feature_importances_.mean()
print('mean:', mean)
importances_feature = list(sorted(zip(model.feature_names_, model.feature_importances_), key = lambda x: x[1], reverse=True))
print(importances_feature)
print()
importances_mean = list(filter(lambda ele: ele[1] < mean, importances_feature))
print(len(importances_mean))
drop_columns = [ele[0] for ele in importances_mean]
print(drop_columns)

importance_train_df = train_input_df.drop(columns = drop_columns)
dp.save_df_to_csv(importance_train_df, "processed/train_input_importance")

test_input_df = dp.read_csv("processed/test_input")
importance_test_df = train_input_df.drop(columns = drop_columns)
dp.save_df_to_csv(importance_test_df, "processed/test_input_importance")