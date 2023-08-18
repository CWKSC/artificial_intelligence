import data_processing as dp

dp.init(__file__)

train_df = dp.read_csv("data/train")

train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

dp.transform(
    train_df,
    ('PassengerId'  , dp.Drop()),
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

dp.transformAll(train_df, dp.Apply(float), except_columns = [])
dp.analysis(train_df)

target_df, input_df = dp.spilt_df(train_df, columns = ['Survived'])
dp.save_df_to_csv(target_df, 'processed/train_target')
dp.save_df_to_csv(input_df, 'processed/train_input')

