from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
import data_processing as dp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from ray import tune
dp.init(__file__)

train_df = dp.read_csv("processed/train")
test_df = dp.read_csv("processed/test")

target_df, input_df = dp.spilt_df(train_df, ['Transported'])
result_df, test_df = dp.spilt_df(test_df, ['PassengerId'])

inputs = input_df
targets =  target_df['Transported'].tolist()
tests = test_df

model = GradientBoostingClassifier()


def objective(config):
    model.set_params(**config)
    model.fit(inputs, targets)
    return {"score": model.score(inputs, targets)}


search_space = {  # ②
    "learning_rate": tune.grid_search([0.001, 0.01, 0.1, 1.0])
}

tuner = tune.Tuner(objective, param_space=search_space)  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)
