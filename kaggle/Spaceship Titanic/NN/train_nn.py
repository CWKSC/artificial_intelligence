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
# 4 2 | 0.792707 | acc 0.79354 loss 0.77858 
# 6 2 82 | 0.779478 3489.563040 | acc 0.76408 loss 0.76268
# 6 2 50 | 0.782009 3481.529834 | acc 0.78512 loss 0.78115
# 512 2 51 | 0.813413 3268.533791 | acc loss
# 17 1 31 | 0.793972 3368.600885 | acc 0.77297

model = BridgeNet(17, 1, 17, 1)

ann.train(
    model,
    input_tensors,
    target_tensors,
    repeat=1000,
    correct_func=ann.compare_float_true_false,
    save_file_name='NN'
)

# predict_list = ann.predict(
#     model,
#     test_tensors
# )

# result_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
# dp.save_df_to_csv(result_df, 'submission/NN')

