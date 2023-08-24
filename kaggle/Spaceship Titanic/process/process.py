import data_processing as dp
import data_analysis as da
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')

def process(df: pd.DataFrame) -> pd.DataFrame:

    df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0])

    groups = df['PassengerId'].apply(lambda x: x.split('_')[0]).value_counts().to_dict()
    df['GroupSize'] = df['PassengerId'].apply(lambda x: groups[x.split('_')[0]])

    df[['Deck', 'CabinNumber','CabinPosition']] = df['Cabin'].str.split('/', expand = True)

    df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand = True)

    dp.transform(
        df,
        ('HomePlanet'  , dp.FillNaWithMode(), dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('CryoSleep'   , dp.FillNaWithMode(), dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('Cabin'       , dp.Drop()),
        ('Destination' , dp.FillNaWithMode(), dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('Age'         , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('VIP'         , dp.FillNaWithMode(), dp.VocabEncode()),
        ('RoomService' , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('FoodCourt'   , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('ShoppingMall', dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('Spa'         , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('VRDeck'      , dp.FillNaWithMean(), dp.Standardize(), dp.Normalize()),
        ('Name'        , dp.Drop()),
        ('GroupId'     , dp.VocabEncode(), dp.Normalize()),
        ('GroupSize'    , dp.Standardize(), dp.Normalize()),
        ('Deck'          , dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('CabinNumber'   , dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('CabinPosition' , dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('FirstName'     , dp.VocabEncode(), dp.Standardize(), dp.Normalize()),
        ('LastName'      , dp.VocabEncode(), dp.Standardize(), dp.Normalize())
    )

    dp.transformAll(test_df, dp.Apply(float), except_columns = ['PassengerId'])
    return df

train_df = process(train_df)
test_df = process(test_df)

dp.transform(train_df, ('Transported', dp.Replace({'True': 1, 'False': 0})))
da.analysis_df(train_df)
da.analysis_df(test_df)

train_df.drop(columns=['PassengerId'], inplace=True)
dp.save_df_to_csv(train_df, '../processed/train')
target_df, input_df = dp.spilt_df(train_df, columns = ['Transported'])
dp.save_df_to_csv(target_df, '../processed/train_target')
dp.save_df_to_csv(input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_input_df, '../processed/test_input')
dp.save_df_to_csv(test_target_df, '../processed/test_target')
