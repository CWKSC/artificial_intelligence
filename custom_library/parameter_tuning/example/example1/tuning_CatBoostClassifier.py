import data_processing as dp
import parameter_tuning as pt
dp.init(__file__)

train_df = dp.read_csv('data/train')

max_args = pt.bayesian_optimization_tuning(
    train_df,
    target_field = 'Survived',
    
    model_setter_func = pt.basic_model_setter_CatBoostClassifier,
    pbounds = pt.basic_pbounds_CatBoostClassifier,

    cross_validation_mode = 'random_sample',
    k_fold_n = 10,
    random_sample_num=10,

    probe_params= {
        'iterations': 402,
        'learning_rate': 0.03604
    }
)
print(max_args)
