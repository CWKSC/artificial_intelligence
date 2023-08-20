import data_analysis as da
import data_processing as dp
dp.init(__file__)


da.analysis_train_test(
    data_dir_path = 'data',
    id_field = 'PassengerId',
    target_field = 'Survived'
)
