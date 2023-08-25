from sklearn.neighbors import KNeighborsClassifier
import data_processing as dp
dp.init(__file__)

input_df = dp.read_csv("processed/train_input").to_numpy()
target_df = dp.read_csv("processed/train_target").to_numpy().ravel()
test_input_df = dp.read_csv("processed/test_input").to_numpy()
test_target_df = dp.read_csv("processed/test_target")

# 5 | 0.71358
# 7 | 0.72924
# 12 | 0.74
# 16 | 
# 32 | 
# 64 | 0.74304

model = KNeighborsClassifier(
    n_neighbors = 32
)
model.fit(input_df, target_df)
# print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)

test_target_df['Transported'] = ['True' if pred > 0.5 else 'False' for pred in predictions]
print(test_target_df['Transported'].value_counts())
dp.save_df_to_csv(test_target_df, "submission/KNeighborsClassifier")
