from catboost import CatBoostClassifier
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 250 0.001 -> 0.79186
# 250 0.001 6 3.0 -> 0.79186
# 250 0.001 6 2.0 -> 0.78229
# 250 0.001 6 4.0 -> 0.79186
# 250 0.001 6 8.0 -> 0.78947
# 250 0.001 12 4.0 -> 0.77511
# 250 0.001 3 4.0 -> 0.78229
# 250 0.001 5 4.0 -> 0.78708
# 250 0.001 7 4.0 -> 0.78468
# 200 0.001 6 4.0 -> 0.79186
# 200 0.0005 6 4.0 -> 0.78947
# 

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.0005,
    depth = 6,
    l2_leaf_reg = 4.0
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Survived'] = predictions.astype(int)
print(test_target_df)
dp.save_df_to_csv(test_target_df, "submission/CatBoostClassifier")
