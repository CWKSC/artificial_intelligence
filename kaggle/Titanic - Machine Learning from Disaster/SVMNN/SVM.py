import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

import data_processing as dp
dp.init(__file__)

inputs = dp.read_csv("../processed/train_input").to_numpy()
targets = dp.read_csv("../processed/train_target").to_numpy().ravel()
test_inputs = dp.read_csv("../processed/test_input").to_numpy()
test_target_df = dp.read_csv("../processed/test_target")

# 0.78468

clf = svm.SVC()
clf.fit(inputs, targets)
predictions = clf.predict(test_inputs)

test_target_df['Survived'] = predictions.astype(int)
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, "../submission/SVM")
