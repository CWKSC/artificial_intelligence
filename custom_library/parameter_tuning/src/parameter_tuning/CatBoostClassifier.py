from catboost import CatBoostClassifier

basic_pbounds_CatBoostClassifier: dict[str, tuple[float, float]] = {
    'iterations': (1, 512),
    'learning_rate': (1e-5, 0.2)
}

def basic_model_setter_CatBoostClassifier(**args):
    return CatBoostClassifier(
        iterations = int(args['iterations']),
        learning_rate = args['learning_rate'],
        verbose = False
    )
