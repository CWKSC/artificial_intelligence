from sklearn.ensemble import GradientBoostingClassifier
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 1000 1 -> 0.70813
# 1000 0.1 -> 0.72966
# 1000 0.001 -> 0.76315
# 1000 0.0001 -> 0.622
# 10000 0.0001 -> 0.76315

# 491 0.06855 -> 0.73444

model = GradientBoostingClassifier(
    n_estimators = 491,
    learning_rate = 0.06855,
    verbose = 1
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Survived'] = predictions.astype(int)
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/GradientBoostingClassifier")
