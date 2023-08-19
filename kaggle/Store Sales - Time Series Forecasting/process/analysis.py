import data_processing as dp
dp.init(__file__)

# dp.analysis(dp.read_csv("data/train"))
# dp.analysis(dp.read_csv("data/test"))
dp.analysis(dp.read_csv("../data/stores"))
dp.analysis(dp.read_csv("../data/transactions"))

# dp.analysis(dp.read_csv("processed/train"))
# dp.analysis(dp.read_csv("processed/test"))
