import data_processing as dp
import artificial_neural_network as ann
from artificial_neural_network import BridgeNet
dp.init(__file__)
ann.init(__file__)

test_input_df = dp.read_csv("../processed/test_input")
test_target_df = dp.read_csv("../processed/test_target")

test_input_tensors = dp.df_to_2d_tensor(test_input_df).to(ann.device)


num_neuron = 17
num_layer = 1

model = ann.load_model(BridgeNet(17, 1, num_neuron, num_layer), 'model/NN_loss')
predict_list = ann.predict(
    model,
    test_input_tensors
)

test_target_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
print(test_target_df['Transported'].value_counts())
dp.save_df_to_csv(test_target_df, '../submission/NN_loss')


model = ann.load_model(BridgeNet(17, 1, num_neuron, num_layer), 'model/NN_accuracy')
predict_list = ann.predict(
    model,
    test_input_tensors
)

test_target_df['Transported'] = ['True' if ele.item() > 0.5 else 'False' for ele in predict_list]
print(test_target_df['Transported'].value_counts())
dp.save_df_to_csv(test_target_df, '../submission/NN_accuracy')