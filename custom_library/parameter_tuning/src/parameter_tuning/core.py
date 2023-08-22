from typing import Any, Callable, Literal, Union

import pandas as pd
from bayes_opt import BayesianOptimization

import data_processing as dp
import cross_validation as cv


def common_model_fit_func(model, input: pd.DataFrame, target: pd.DataFrame):
    model.fit(input, target.to_numpy().ravel())

def common_model_score_func(model, input: pd.DataFrame, target: pd.DataFrame):
    return model.score(input, target)

def bayesian_optimization_tuning(
    df: pd.DataFrame,
    target_field: str,
    model_setter_func: Callable,
    pbounds: dict[str, tuple[float, float]],
    optimizer_init_points: int = 5,
    optimizer_n_iter: int = 25,
    fit_func: Callable = common_model_fit_func,
    score_func: Callable = common_model_score_func,
    cross_validation_mode: Union[Literal['k_fold'], Literal['random_sample'], None] = None,
    k_fold_n: int = 5,
    random_sample_valid_ratio: float = 0.5,
    random_sample_num: int = 30,
    probe_params: dict = None
) -> dict:
    
    if cross_validation_mode == None:

        train_target, train_input = dp.spilt_df(df, columns=[target_field])

        def model_optimization_func(**args):
            model = model_setter_func(**args)
            fit_func(model, train_input, train_target)
            return score_func(model, train_input, train_target)
        
        optimizer = BayesianOptimization(
            f = model_optimization_func,
            pbounds = pbounds
        )
        if probe_params != None:
            optimizer.probe(
                params= probe_params,
                lazy = True
            )
        optimizer.maximize(
            init_points = optimizer_init_points,
            n_iter = optimizer_n_iter,
        )

        return
    
    cross_validation_data: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None

    if cross_validation_mode == 'k_fold':
        cross_validation_data = cv.k_fold(
            df, 
            target_field, 
            n = k_fold_n
        )
        cross_validation_n = k_fold_n
    
    elif cross_validation_mode == 'random_sample':
        cross_validation_data = cv.random_sampling(
            df, 
            target_field, 
            valid_ratio = random_sample_valid_ratio, 
            n = random_sample_num
        )
        cross_validation_n = random_sample_num

    def model_optimization_func(**args):
        score_sum = 0
        for train_input, train_target, valid_input, valid_target in cross_validation_data:
            model = model_setter_func(**args)
            fit_func(model, train_input, train_target)
            score_sum += score_func(model, valid_input, valid_target)
        return score_sum / cross_validation_n

    optimizer = BayesianOptimization(
        f = model_optimization_func,
        pbounds = pbounds
    )
    if probe_params != None:
        optimizer.probe(
            params= probe_params,
            lazy = True
        )
    optimizer.maximize(
        init_points = optimizer_init_points,
        n_iter = optimizer_n_iter,
    )

    return optimizer.max


