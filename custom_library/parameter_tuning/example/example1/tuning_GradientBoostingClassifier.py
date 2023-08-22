import data_processing as dp
import parameter_tuning as pt
dp.init(__file__)

train_df = dp.read_csv('data/train')

max_args = pt.bayesian_optimization_tuning(
    train_df,
    target_field = 'Survived',
    model_setter_func = pt.basic_model_setter_GradientBoostingClassifier,
    pbounds = pt.basic_pbounds_GradientBoostingClassifier,

    cross_validation_mode = 'random_sample',
    k_fold_n = 10,
    random_sample_num=5,

    optimizer_n_iter=50,
    probe_params={
        'n_estimators': 199,
        'learning_rate': 0.05439
    }
)
print(max_args)
