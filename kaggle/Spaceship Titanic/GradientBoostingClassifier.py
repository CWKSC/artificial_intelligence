from sklearn.ensemble import GradientBoostingClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input_importance")
target_df = dp.read_csv("processed/train_target")
test_input_df = dp.read_csv("processed/test_input_importance")
test_target_df = dp.read_csv("processed/test_target")

model = GradientBoostingClassifier()
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)
print(predictions)
predictions = ['True' if pred else 'False' for pred in predictions]

test_target_df['Transported'] = predictions
dp.save_df_to_csv(test_target_df, "submission/GradientBoostingClassifier")
