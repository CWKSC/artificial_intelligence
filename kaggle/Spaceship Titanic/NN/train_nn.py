import data_processing as dp
import artificial_neural_network as ann
from artificial_neural_network import BridgeNet
dp.init(__file__)
ann.init(__file__)

input_df = dp.read_csv("../processed/train_input")
target_df = dp.read_csv("../processed/train_target")

input_tensors = dp.df_to_2d_tensor(input_df)
target_tensors = dp.df_to_2d_tensor(target_df)

# 3 2 | 0.795928 | loss 0.79401
# 4 2 | 0.792707 | loss 0.77858 | acc 
# 

model = BridgeNet(17, 1, 4, 2)

ann.train(
    model,
    input_tensors,
    target_tensors,
    repeat=200,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN'
)

# predict_list = ann.predict(
#     model,
#     test_tensors
# )

# result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
# dp.save_df_to_csv(result_df, 'submission/NN')

