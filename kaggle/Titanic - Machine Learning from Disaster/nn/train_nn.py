import data_processing as dp
import artificial_neural_network as ann
import torch
from artificial_neural_network import BridgeNet
dp.init(__file__)
ann.init(__file__)

input_df = dp.read_csv("../processed/train_input")
target_df = dp.read_csv("../processed/train_target")

# train_target_2d_list = dp.df_to_2d_list(target_df)
# train_target_2d_list = [[0, 1] if ele[0] > 0.5 else [1, 0] for ele in train_target_2d_list]
# train_target_tensor = torch.tensor(train_target_2d_list, dtype=torch.float64)
# print(train_target_tensor.shape)

input_tensors = dp.df_to_2d_tensor(input_df)
target_tensors = dp.df_to_2d_tensor(target_df)
print(target_tensors.shape)
print(input_tensors.shape)

model = BridgeNet(10, 1, 10, 2)

# 30 7 15 | 0.832772 | 0.78229
# 30 7 5 | 0.800224 | 0.7799
# 3 2 30 | 0.817059 381.976152 | 0.7799
# 6 2 30 | 0.815937 378.627133 | acc 0.78708 loss 0.79425
# 6 2 30 | 0.818182 375.966702 | acc 0.78229 loss
# 

ann.train(
    model,
    input_tensors,
    target_tensors,
    repeat = 30,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN'
)
