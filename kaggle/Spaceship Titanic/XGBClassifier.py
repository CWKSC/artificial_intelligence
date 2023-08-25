from xgboost import XGBClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input")
target_df = dp.read_csv("processed/train_target").to_numpy().ravel()
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

# 0.7753

model = XGBClassifier()
model.fit(input_df, target_df)
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)
predictions = ['True' if pred > 0.5 else 'False' for pred in predictions]

test_target_df['Transported'] = predictions
dp.save_df_to_csv(test_target_df, "submission/XGBClassifier")

