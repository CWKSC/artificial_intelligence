from sklearn.neighbors import KNeighborsClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input").to_numpy()
target_df = dp.read_csv("processed/train_target").to_numpy().ravel()
test_input_df = dp.read_csv("processed/test_input").to_numpy()
test_target_df = dp.read_csv("processed/test_target")

# 5 | 0.75837
# 6 | 0.76315
# 7 |
# 10 | 0.7799
# 16 | 0.77511

model = KNeighborsClassifier(
    n_neighbors = 16
)
model.fit(input_df, target_df)
# print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Survived'] = predictions.astype(int)
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/KNeighborsClassifier")
