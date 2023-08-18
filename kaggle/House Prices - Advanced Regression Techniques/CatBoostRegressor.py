from catboost import CatBoostRegressor
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# grid = {'iterations': [220, 200, 180],
#         'learning_rate': [0.15, 0.1],
#         'depth': [4, 6, 8],
#         'l2_leaf_reg': [0.8, 1, 2]}

# model = CatBoostRegressor()
# grid_search_result = model.grid_search(grid, input_df, target_df.to_numpy().ravel(), verbose=100)
# print(grid_search_result)

model = CatBoostRegressor(
    depth=6,
    iterations=220,
    learning_rate=0.2,
    l2_leaf_reg=0.6
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['SalePrice'] = predictions
print(test_target_df['SalePrice'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostRegressor")