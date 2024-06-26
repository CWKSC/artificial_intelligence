
import data_processing as dp
dp.init(__file__)

test_df = dp.read_csv('../data/test')

dp.transform(
    test_df,
    ('Id'             ),
    ('MSSubClass'     ),
    ('MSZoning'       , dp.VocabEncode()),
    ('LotFrontage'    , dp.FillNa(-1)),
    ('LotArea'        ),
    ('Street'         , dp.VocabEncode()),
    ('Alley'          , dp.VocabEncode()),
    ('LotShape'       , dp.VocabEncode()),
    ('LandContour'    , dp.VocabEncode()),
    ('Utilities'      , dp.VocabEncode()),
    ('LotConfig'      , dp.VocabEncode()),
    ('LandSlope'      , dp.VocabEncode()),
    ('Neighborhood'   , dp.VocabEncode()),
    ('Condition1'     , dp.VocabEncode()),
    ('Condition2'     , dp.VocabEncode()),
    ('BldgType'       , dp.VocabEncode()),
    ('HouseStyle'     , dp.VocabEncode()),
    ('OverallQual'    ),
    ('OverallCond'    ),
    ('YearBuilt'      ),
    ('YearRemodAdd'   ),
    ('RoofStyle'      , dp.VocabEncode()),
    ('RoofMatl'       , dp.VocabEncode()),
    ('Exterior1st'    , dp.VocabEncode()),
    ('Exterior2nd'    , dp.VocabEncode()),
    ('MasVnrType'     , dp.VocabEncode()),
    ('MasVnrArea'     , dp.FillNa(-1)),
    ('ExterQual'      , dp.VocabEncode()),
    ('ExterCond'      , dp.VocabEncode()),
    ('Foundation'     , dp.VocabEncode()),
    ('BsmtQual'       , dp.VocabEncode()),
    ('BsmtCond'       , dp.VocabEncode()),
    ('BsmtExposure'   , dp.VocabEncode()),
    ('BsmtFinType1'   , dp.VocabEncode()),
    ('BsmtFinSF1'     , dp.FillNa(-1)),
    ('BsmtFinType2'   , dp.VocabEncode()),
    ('BsmtFinSF2'     , dp.FillNa(-1)),
    ('BsmtUnfSF'      , dp.FillNa(-1)),
    ('TotalBsmtSF'    , dp.FillNa(-1)),
    ('Heating'        , dp.VocabEncode()),
    ('HeatingQC'      , dp.VocabEncode()),
    ('CentralAir'     , dp.VocabEncode()),
    ('Electrical'     , dp.VocabEncode()),
    ('1stFlrSF'       ),
    ('2ndFlrSF'       ),
    ('LowQualFinSF'   ),
    ('GrLivArea'      ),
    ('BsmtFullBath'   , dp.FillNa(-1)),
    ('BsmtHalfBath'   , dp.FillNa(-1)),
    ('FullBath'       ),
    ('HalfBath'       ),
    ('BedroomAbvGr'   ),
    ('KitchenAbvGr'   ),
    ('KitchenQual'    , dp.VocabEncode()),
    ('TotRmsAbvGrd'   ),
    ('Functional'     , dp.VocabEncode()),
    ('Fireplaces'     ),
    ('FireplaceQu'    , dp.VocabEncode()),
    ('GarageType'     , dp.VocabEncode()),
    ('GarageYrBlt'    , dp.FillNa(-1)),
    ('GarageFinish'   , dp.VocabEncode()),
    ('GarageCars'     , dp.FillNa(-1)),
    ('GarageArea'     , dp.FillNa(-1)),
    ('GarageQual'     , dp.VocabEncode()),
    ('GarageCond'     , dp.VocabEncode()),
    ('PavedDrive'     , dp.VocabEncode()),
    ('WoodDeckSF'     ),
    ('OpenPorchSF'    ),
    ('EnclosedPorch'  ),
    ('3SsnPorch'      ),
    ('ScreenPorch'    ),
    ('PoolArea'       ),
    ('PoolQC'         , dp.VocabEncode()),
    ('Fence'          , dp.VocabEncode()),
    ('MiscFeature'    , dp.VocabEncode()),
    ('MiscVal'        ),
    ('MoSold'         ),
    ('YrSold'         ),
    ('SaleType'       , dp.VocabEncode()),
    ('SaleCondition'  , dp.VocabEncode()),
)
dp.transformAll(test_df, dp.Apply(float), except_columns = ['Id'])

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['Id'])
dp.save_df_to_csv(test_target_df, 'processed/test_target')
dp.save_df_to_csv(test_input_df, 'processed/test_input')

dp.analysis(test_df)
print(test_target_df.head(10))
print(test_input_df.head(10))

dp.save_df_to_csv(test_df, 'processed/test')