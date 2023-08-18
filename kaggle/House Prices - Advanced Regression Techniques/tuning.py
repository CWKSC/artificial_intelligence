import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn import cross_validation, metrics
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor 
import data_processing as dp
dp.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

model = GradientBoostingRegressor()


param_n_estimators = {'n_estimators': range(143,147,1) }
gsearch1 = GridSearchCV(estimator = model, 
                       param_grid = param_n_estimators)
gsearch1.fit(input_df, target_df.to_numpy().ravel())
print(gsearch1.best_params_)
print(gsearch1.best_score_)

model.set_params(gsearch1.best_params_)

param_max_depth = {'max_depth':range(2, 4, 1)}
gsearch1 = GridSearchCV(estimator = model, 
                       param_grid = param_max_depth)
gsearch1.fit(input_df, target_df.to_numpy().ravel())
print(gsearch1.best_params_)
print(gsearch1.best_score_)




