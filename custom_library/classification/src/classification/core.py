from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from ngboost import NGBClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import pandas as pd

import cross_validation as cv

common_classifiers = [
    ('CatBoostClassifier', CatBoostClassifier(verbose=False)),
    ('LGBMClassifier', LGBMClassifier(force_row_wise=True)),
    ('XGBClassifier', XGBClassifier()),
    ('XGBRFClassifier', XGBRFClassifier()),
    ('NGBClassifier', NGBClassifier()),
    ('GradientBoostingClassifier', GradientBoostingClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('BaggingClassifier', BaggingClassifier()),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('HistGradientBoostingClassifier', HistGradientBoostingClassifier()),
    ('ExtraTreesClassifier', ExtraTreesClassifier()),
    ('KNeighborsClassifier n_neighbors = 2', KNeighborsClassifier(n_neighbors = 2)),
    ('KNeighborsClassifier n_neighbors = 4', KNeighborsClassifier(n_neighbors = 4)),
    ('KNeighborsClassifier n_neighbors = 8', KNeighborsClassifier(n_neighbors = 8)),
    ('KNeighborsClassifier n_neighbors = 16', KNeighborsClassifier(n_neighbors = 16)),
    ('KNeighborsClassifier n_neighbors = 32', KNeighborsClassifier(n_neighbors = 32)),
    ('KNeighborsClassifier n_neighbors = 64', KNeighborsClassifier(n_neighbors = 64)),
    ('KNeighborsClassifier n_neighbors = 128', KNeighborsClassifier(n_neighbors = 128)),
    ('KNeighborsClassifier n_neighbors = 256', KNeighborsClassifier(n_neighbors = 256)),
]

def try_classifier(
    train_df: pd.DataFrame,
    valid_ratio = 0.5,
    n: int = 3
):
    random_sample_list = cv.random_sampling(train_df, 'Survived', valid_ratio = valid_ratio, n = n)

    result = {}

    for name, classifier in common_classifiers:
        result[name] = {}
        print(name)
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        for i, (train_input, train_target, valid_input, valid_target) in enumerate(random_sample_list):
            print(i)
            classifier.fit(train_input.to_numpy(), train_target.to_numpy().ravel())
            train_accuracy = classifier.score(train_input.to_numpy(), train_target.to_numpy().ravel())
            valid_accuracy = classifier.score(valid_input.to_numpy(), valid_target.to_numpy().ravel())       
            print('train', train_accuracy)
            print('valid', valid_accuracy)
            result[name][f'train_accuracy_{i}'] = train_accuracy
            result[name][f'valid_accuracy_{i}'] = valid_accuracy
            train_accuracy_sum += train_accuracy
            valid_accuracy_sum += valid_accuracy
        train_accuracy_mean = train_accuracy_sum / n
        valid_accuracy_mean = valid_accuracy_sum / n
        print('train_accuracy_sum', train_accuracy_sum)
        print('valid_accuracy_sum', valid_accuracy_sum)
        print('train_accuracy_mean', train_accuracy_mean)
        print('valid_accuracy_mean', valid_accuracy_mean)
        result[name]['train_accuracy_sum'] = train_accuracy_sum
        result[name]['valid_accuracy_sum'] = valid_accuracy_sum
        result[name]['train_accuracy_mean'] = train_accuracy_mean
        result[name]['valid_accuracy_mean'] = valid_accuracy_mean

    result_df = pd.DataFrame()

    names = []
    train_accuracy_mean = []
    valid_accuracy_mean = []
    diff = []
    for name in result:
        names.append(name)
        train_accuracy_mean.append(result[name]['train_accuracy_mean'])
        valid_accuracy_mean.append(result[name]['valid_accuracy_mean'])
        diff.append(result[name]['train_accuracy_mean'] - result[name]['valid_accuracy_mean'])
    result_df['classifier'] = names
    result_df['train_accuracy_mean'] = train_accuracy_mean
    result_df['valid_accuracy_mean'] = valid_accuracy_mean
    result_df['diff'] = diff

    return result_df.sort_values(by=['valid_accuracy_mean'], ascending=False)