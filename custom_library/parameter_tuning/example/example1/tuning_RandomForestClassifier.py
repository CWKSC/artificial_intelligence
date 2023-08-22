import data_processing as dp
import parameter_tuning as pt
dp.init(__file__)

train_df = dp.read_csv('data/train')

max_args = pt.bayesian_optimization_tuning(
    train_df,
    target_field = 'Survived',
    model_setter_func = pt.basic_model_setter_RandomForestClassifier,
    pbounds = pt.basic_pbounds_RandomForestClassifier,
    cross_validation_mode = 'k_fold',
    k_fold_n = 10
)
print(max_args)
