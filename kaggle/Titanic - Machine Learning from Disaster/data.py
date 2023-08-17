import data_processing as dp

dp.init(__file__)

train_dataframe = dp.read_csv("data/train")
test_dataframe = dp.read_csv("data/test")

dp.transform(
    train_dataframe,
    ('PassengerId', dp.Drop()),
    ('Name',        dp.Apply(len)),
    ('Sex',         dp.VocabEncode()),
    ('Age',         dp.FillNa(-1)),
    ('Ticket',      dp.VocabEncode()),
    ('Cabin',       dp.VocabEncode()),
    ('Embarked',    dp.VocabEncode())
)
dp.transformAll(train_dataframe, dp.Apply(float))
print(train_dataframe)
dp.save_df_to_csv(train_dataframe, 'processed/train')

dp.transform(
    test_dataframe,
    ('Name',        dp.Apply(len)),
    ('Sex',         dp.VocabEncode()),
    ('Age',         dp.FillNa(-1)),
    ('Ticket',      dp.VocabEncode()),
    ('Fare',        dp.FillNa(-1)),
    ('Cabin',       dp.VocabEncode()),
    ('Embarked',    dp.VocabEncode())
)
dp.transformAll(test_dataframe, dp.Apply(float))
print(test_dataframe)
dp.save_df_to_csv(test_dataframe, 'processed/test')

print(dp.toTensors(train_dataframe))
print(dp.toTensors(test_dataframe))