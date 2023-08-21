from sklearn.ensemble import GradientBoostingClassifier
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

model = GradientBoostingClassifier()
model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Survived'] = predictions.astype(int)
print(test_target_df)
dp.save_df_to_csv(test_target_df, "submission/GradientBoostingClassifier")
