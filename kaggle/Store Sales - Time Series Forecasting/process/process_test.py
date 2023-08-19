import data_processing as dp
import numpy as np
import pandas as pd
dp.init(__file__)

test_df = dp.read_csv('../data/test')

test_df['date'] = pd.to_datetime(test_df['date'])    
test_df['date'] = (test_df['date'] - pd.Timestamp('2013-01-01'))  / np.timedelta64(1,'D')

dp.transform(
    test_df,
    ('id'           ),
    ('date'         ),
    ('store_nbr'    ),
    ('family'       , dp.VocabEncode()),
    ('onpromotion'  ),
)
dp.transformAll(test_df, dp.Apply(float), except_columns = ['id'])

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['id'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')

dp.analysis(test_df)
print(test_target_df.head(10))
print(test_input_df.head(10))