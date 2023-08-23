import data_processing as dp
import artificial_neural_network as ann
from Model import Model
dp.init(__file__)
ann.init(__file__)

target_df = dp.read_csv("processed/train_target")
input_df = dp.read_csv("processed/train_input")
test_input_df = dp.read_csv("processed/test_input")
test_target_df = dp.read_csv("processed/test_target")

target_tensors = dp.df_to_2d_tensor(target_df)
input_tensors = dp.df_to_2d_tensor(input_df)
test_input_tensors = dp.df_to_2d_tensor(test_input_df)

model = Model()

ann.train(
    model,
    target_tensors,
    input_tensors,
    correct_func=ann.compare_float_isclose,
    save_file_name='NN'
)
