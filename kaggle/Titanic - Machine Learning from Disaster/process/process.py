import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv("data/train")
test_df = dp.read_csv("./data/test")

def process(df):

    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    dp.transform(
        df,
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

    dp.transformAll(df, dp.Apply(float), except_columns = [])

    return df

train_df = process(train_df)
test_df = process(test_df)

target_df, input_df = dp.spilt_df(train_df, columns = ['Survived'])
dp.save_df_to_csv(target_df, 'processed/train_target')
dp.save_df_to_csv(input_df, 'processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_input_df, 'processed/test_input')
dp.save_df_to_csv(test_target_df, 'processed/test_target')

