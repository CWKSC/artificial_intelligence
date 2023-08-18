import data_processing as dp
dp.init(__file__)

test_input_df = dp.read_csv("data/test")

test_input_df['Title'] = test_input_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

dp.transform(
    test_input_df,
    ('PassengerId'  ),
    ('Pclass'       ),
    ('Name'         , dp.Drop()),
    ('Sex'          , dp.VocabEncode()),
    ('Age'          , dp.FillNaWithMean()),
    ('SibSp'        ),
    ('Parch'        ),
    ('Ticket'       , dp.VocabEncode()),
    ('Fare'         , dp.FillNaWithMean()),
    ('Cabin'        , dp.VocabEncode()),
    ('Embarked'     , dp.FillNaWithMode(), dp.VocabEncode()),
    ('Title'        , dp.VocabEncode()),
)
dp.transformAll(test_input_df, dp.Apply(float), except_columns = ['PassengerId'])
dp.analysis(test_input_df)

test_target_df, test_input_df = dp.spilt_df(test_input_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_input_df, 'processed/test_input')
dp.save_df_to_csv(test_target_df, 'processed/test_target')
