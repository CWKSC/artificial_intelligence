import data_processing as dp
import parameter_tuning as pt
import artificial_neural_network as ann
ann.init(__file__)
dp.init(__file__)

train_df = dp.read_csv('data/train')

def fit_func(model, input, target):
    ann.train(
        model,
        dp.toTensors(input),
        dp.toTensors(target),
        repeat=32,
        correct_func = ann.compare_float_true_false,
        save_mode = None,
        ctrl_c_skip = False,
        verbose = False
    )

def score_func(model, input, target):
    return ann.eval(
        model,
        dp.toTensors(input),
        dp.toTensors(target),
        correct_func=ann.compare_float_true_false,
        verbose = False
    )

max_args = pt.bayesian_optimization_tuning(
    train_df,
    target_field = 'Survived',
    model_setter_func = pt.basic_model_setter_BridgeNet_creator(10, 1),
    fit_func = fit_func,
    score_func = score_func,
    pbounds = pt.basic_pbounds_BridgeNet,
    optimizer_init_points = 5,
    optimizer_n_iter = 1000,

    cross_validation_mode = None,
    k_fold_n = 3, 
    random_sample_num = 3,
    random_sample_valid_ratio = 0.9,

    probe_params= {
        'num_neuron': 233,
        'num_layer': 1.0
    }
)
print(max_args)
