from sklearn.ensemble import RandomForestClassifier
import data_processing as dp
import pandas as pd
dp.init(__file__)

train_dataframe = dp.read_csv("processed/train")
test_dataframe = dp.read_csv("processed/test")


features = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
y = train_dataframe['Survived']
X = pd.get_dummies(train_dataframe[features])
X_test =  pd.get_dummies(test_dataframe[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
print(predictions)
predictions = list(map(int, predictions))

output = pd.DataFrame({'PassengerId': test_dataframe.PassengerId.apply(int), 'Survived': predictions})
dp.save_df_to_csv(output, "submission/RandomForest")
