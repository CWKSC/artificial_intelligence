from catboost import CatBoostClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input")
target_df = dp.read_csv("processed/train_target")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 250 0.001 -> 0.78676
# 500 0.001 -> 0.78933
# 256 0.01 -> 0.79962
# 512 0.01 -> 0.80149
# 1024 0.01 -> 0.80102
# 512 0.02 -> 0.79939
# 512 0.005 -> 0.79705
# 1024 0.005 -> 0.80126
# 2048 0.005 -> 0.80009
# 256 0.1 -> 0.79354

# After data process
# 512 0.01 -> 0.79775
# 256 0.01 -> 0.79752
# 1024 0.01 -> 0.80266
# 2048 0.01 -> 0.79705
# 1536 0.01 -> 0.80079
# 768 0.01 -> 0.80009
# 1280 0.01 -> 0.80243
# 1152 0.01 -> 0.80219
# 1216 0.01 -> 0.80219
# 1088 0.01 -> 0.80266

# 1024 0.005 -> 0.79658
# 1536 0.005 -> 0.79728
# 768 0.005 -> 0.79728

# 512 0.02 -> 0.79822
# 1024 0.02 -> 0.79284

model = CatBoostClassifier(
    iterations = 1088,
    learning_rate = 0.01,
    verbose = 100
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Transported'] = ['True' if pred > 0.5 else 'False' for pred in predictions]
print(test_target_df['Transported'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostClassifier")