import data_processing as dp
import numpy as np
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')
stores_df = dp.read_csv('../data/stores')
oil_df = dp.read_csv('../data/oil')
transactions_df = dp.read_csv('../data/transactions')
holidays_events_df = dp.read_csv('../data/holidays_events')

oil_df = oil_df.dropna()
# oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(method='bfill')

def process(df):
    df = pd.merge(df, stores_df, on='store_nbr')
    print(df)
    
    df = pd.merge(df, oil_df, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].fillna(method='bfill')
    print(df)

    df = pd.merge(df, holidays_events_df, on='date', how='left')
    print(df)

    df['date'] = pd.to_datetime(df['date'])  
    df['date'] = (df['date'] - pd.Timestamp('2013-01-01'))  / np.timedelta64(1,'D')

    dp.transform(
        df,
        ('family'        , dp.VocabEncode()),
        ('city'          , dp.VocabEncode()),
        ('state'         , dp.VocabEncode()),
        ('type_x'        , dp.VocabEncode()),
        ('type_y'        , dp.VocabEncode()),
        ('locale'        , dp.VocabEncode()),
        ('locale_name'   , dp.VocabEncode()),
        ('description'   , dp.VocabEncode()),
        ('transferred'   , dp.FillNa('False'), dp.Replace({'True': 1, 'False': 0})),
    )
    dp.transformAll(df, dp.Apply(float), except_columns = ['id'])

    print(df)

    return df

train_df = process(train_df)
test_df = process(test_df)

train_df.drop(columns=['id'], axis=1, inplace=True)
train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['sales'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['id'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')

dp.analysis(train_input_df)
dp.analysis(test_input_df)



