import data_processing as dp
import data_analysis as da
dp.init(__file__)

da.create_template()

dp.analysis(dp.read_csv("data/train"))
dp.analysis(dp.read_csv("data/test"))

# dp.analysis(dp.read_csv("processed/train"))
# dp.analysis(dp.read_csv("processed/test"))
