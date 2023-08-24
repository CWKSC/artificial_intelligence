import data_processing as dp
import artificial_neural_network as ann
from artificial_neural_network import BridgeNet
dp.init(__file__)
ann.init(__file__)

test_input_df = dp.read_csv("../processed/test_input")
test_target_df = dp.read_csv("../processed/test_target")

test_input_tensors = dp.df_to_2d_tensor(test_input_df).to(ann.device)

model = ann.load_model(BridgeNet(10, 1, 30, 7), 'model/NN_loss')
predictions = ann.predict(
    model,
    test_input_tensors
)

test_target_df['Survived'] =  [1 if ele.item() > 0.5 else 0 for ele in predictions]
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, '../submission/NN_loss')


model = ann.load_model(BridgeNet(10, 1, 30, 7), 'model/NN_accuracy')
predictions = ann.predict(
    model,
    test_input_tensors
)
test_target_df['Survived'] =  [1 if ele.item() > 0.5 else 0 for ele in predictions]
print(test_target_df['Survived'].value_counts())
dp.save_df_to_csv(test_target_df, '../submission/NN_accuracy')


