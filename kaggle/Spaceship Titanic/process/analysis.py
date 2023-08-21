import data_analysis as da
import data_processing as dp
dp.init(__file__)


da.analysis_train_test(
    data_dir_path = '../data',
    id_field = 'PassengerId',
    target_field = 'Transported'
)

train_input_df = dp.read_csv("../processed/train_input")
test_input_df = dp.read_csv("../processed/test_input")

da.analysis_df(train_input_df)
da.analysis_df(test_input_df)
