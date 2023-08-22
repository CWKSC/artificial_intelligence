
from sklearn.ensemble import GradientBoostingClassifier

basic_pbounds_GradientBoostingClassifier = {
    'n_estimators': (1, 512),
    'learning_rate': (1e-5, 0.2)
}


def basic_model_setter_GradientBoostingClassifier(**args):
    return GradientBoostingClassifier(
        n_estimators = int(args['n_estimators']),
        learning_rate = args['learning_rate']
    )

