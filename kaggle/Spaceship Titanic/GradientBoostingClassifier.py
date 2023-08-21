from sklearn.ensemble import GradientBoostingClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input_importance")
target_df = dp.read_csv("processed/train_target")
test_input_df = dp.read_csv("processed/test_input_importance")
test_target_df = dp.read_csv("processed/test_target")

# 100 0.025 -> 0.79097
# 200 0.025 -> 0.78793
# 50 0.025 -> 0.77016
# 150 0.025 -> 0.79003
# 125 0.025 -> 0.78957
# 200 0.0001 -> 0.72433
# 1000 0.01 -> 0.78536

model = GradientBoostingClassifier(
    n_estimators = 1000,
    learning_rate = 0.01
)
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)
print(predictions)
predictions = ['True' if pred else 'False' for pred in predictions]

test_target_df['Transported'] = predictions
dp.save_df_to_csv(test_target_df, "submission/GradientBoostingClassifier")
