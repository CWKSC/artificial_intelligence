from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

model_gbc = GradientBoostingClassifier()
model_cat = CatBoostClassifier()
model_rfc = RandomForestClassifier()
model_lgbm = LGBMClassifier()
model_xgb = XGBClassifier()

lr = LogisticRegression()

estimators = [
    ('gbc', model_gbc),
    ('cat', model_cat),
    ('rfc', model_rfc),
    ('lgbm', model_lgbm),
    ('xgb', model_xgb),
]
model = StackingClassifier(
    estimators=estimators, 
    final_estimator=lr
)

model.fit(input_df, target_df.to_numpy().ravel())
print(model.score(input_df, target_df))

predictions = model.predict(test_input_df)
predictions = ['True' if pred > 0.5 else 'False' for pred in predictions]

test_target_df['Transported'] = predictions
dp.save_df_to_csv(test_target_df, "submission/StackingClassifier")




