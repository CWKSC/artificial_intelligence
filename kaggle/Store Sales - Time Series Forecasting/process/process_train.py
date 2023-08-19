import data_processing as dp
import numpy as np
import pandas as pd
dp.init(__file__)

train_df = dp.read_csv('../data/train')

train_df['date'] = pd.to_datetime(train_df['date'])  
train_df['date'] = (train_df['date'] - pd.Timestamp('2013-01-01'))  / np.timedelta64(1,'D')

dp.transform(
    train_df,
    ('id'           , dp.Drop()),
    ('date'         ),
    ('store_nbr'    ),
    ('family'       , dp.VocabEncode()),
    ('sales'        ),
    ('onpromotion'  ),
)
dp.transformAll(train_df, dp.Apply(float), except_columns = [])

train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['sales'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

dp.analysis(train_df)
print(train_target_df.head(10))
print(train_input_df.head(10))