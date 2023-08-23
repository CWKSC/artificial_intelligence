import data_processing as dp
import data_analysis as da
dp.init(__file__)

da.analysis_train_test(
    data_dir_path = '../data',
    id_field = 'PassengerId',
    target_field = 'Survived'
)


# dp.analysis(dp.read_csv("processed/train"))
# dp.analysis(dp.read_csv("processed/test"))
