import data_processing as dp
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')

def process(df: pd.DataFrame) -> pd.DataFrame:
    dp.transform(
        df,
        ('HomePlanet'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('CryoSleep'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Cabin'       , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Destination' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Age'         , dp.FillNaWithMean()),
        ('VIP'         , dp.FillNaWithMode(), dp.VocabEncode()),
        ('RoomService' , dp.FillNaWithMean()),
        ('FoodCourt'   , dp.FillNaWithMean()),
        ('ShoppingMall', dp.FillNaWithMean()),
        ('Spa'         , dp.FillNaWithMean()),
        ('VRDeck'      , dp.FillNaWithMean()),
        ('Name'        , dp.FillNaWithMode(), dp.VocabEncode()),
    )
    return df

train_df = process(train_df)
test_df = process(test_df)

train_df.drop(columns=['PassengerId'], inplace=True)
train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['Transported'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['PassengerId'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')
