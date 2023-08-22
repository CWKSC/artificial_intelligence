import data_processing as dp
import data_analysis as da
dp.init(__file__)

da.analysis_train_test('../data', 'Id', 'SalePrice')

da.analysis_df(dp.read_csv('../processed/train_input'))
