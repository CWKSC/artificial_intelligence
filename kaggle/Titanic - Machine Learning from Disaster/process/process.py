
import pandas as pd
import data_analysis as da
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')

def process(df: pd.DataFrame) -> pd.DataFrame:

    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    dp.transform(
        df,
        ('Pclass'  , dp.Normalize()),
        ('Name'    , dp.Drop()),
        ('Sex'     , dp.VocabEncode(), dp.Normalize()),
        ('Age'     , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('SibSp'   , dp.Standardize(), dp.Normalize()),
        ('Parch'   , dp.Standardize(), dp.Normalize()),
        ('Ticket'  , dp.VocabEncode(), dp.Normalize()),
        ('Fare'    , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('Cabin'   , dp.FillNaWithMode(), dp.VocabEncode(), dp.Normalize()),
        ('Embarked', dp.FillNaWithMode(), dp.VocabEncode(), dp.Normalize()),
        ('Title'   , dp.VocabEncode(), dp.Normalize()),
    )
    da.analysis_df(df)
    dp.transformAll(df, dp.Apply(float), except_columns = ['PassengerId'])
    return df

train_df = process(train_df)
test_df = process(test_df)

train_df.drop(columns=['PassengerId'], inplace=True)
dp.save_df_to_csv(train_df, '../processed/train')
train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['Survived'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')
