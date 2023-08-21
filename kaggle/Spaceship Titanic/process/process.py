import data_processing as dp
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
        ('HomePlanet'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('CryoSleep'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Cabin'       , dp.Drop()),
        ('Destination' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Age'         , dp.FillNaWithMean(), dp.Standardize()),
        ('VIP'         , dp.FillNaWithMode(), dp.VocabEncode()),
        ('RoomService' , dp.FillNaWithMean(), dp.Standardize()),
        ('FoodCourt'   , dp.FillNaWithMean(), dp.Standardize()),
        ('ShoppingMall', dp.FillNaWithMean(), dp.Standardize()),
        ('Spa'         , dp.FillNaWithMean(), dp.Standardize()),
        ('VRDeck'      , dp.FillNaWithMean(), dp.Standardize()),
        ('Name'        , dp.Drop()),
        ('Deck'          , dp.VocabEncode()),
        ('CabinNumber'   , dp.VocabEncode()),
        ('CabinPosition' , dp.VocabEncode()),
        ('FirstName'     , dp.VocabEncode()),
        ('LastName'      , dp.VocabEncode())
    )
    dp.transformAll(test_df, dp.Apply(float), except_columns = ['PassengerId'])
    return df

train_df = process(train_df)
test_df = process(test_df)

dp.transform(train_df, ('Transported', dp.Replace({'True': 1, 'False': 0})))
train_df.drop(columns=['PassengerId'], inplace=True)
target_df, input_df = dp.spilt_df(train_df, columns = ['Transported'])
dp.save_df_to_csv(target_df, '../processed/train_target')
dp.save_df_to_csv(input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_input_df, '../processed/test_input')
dp.save_df_to_csv(test_target_df, '../processed/test_target')
