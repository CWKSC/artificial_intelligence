from catboost import CatBoostClassifier
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 250 0.001 -> 0.78676
# 500 0.001 -> 0.78933

model = CatBoostClassifier(
    iterations=375,
    learning_rate=0.001
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Transported'] = ['True' if pred > 0.5 else 'False' for pred in predictions]
print(test_target_df['Transported'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/CatBoostClassifier")