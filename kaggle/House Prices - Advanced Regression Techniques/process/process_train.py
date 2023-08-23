
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv('../data/train')

dp.transform(
    train_df,
    ('Id'             , dp.Drop()),
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
    ('BsmtFinSF1'     ),
    ('BsmtFinType2'   , dp.VocabEncode()),
    ('BsmtFinSF2'     ),
    ('BsmtUnfSF'      ),
    ('TotalBsmtSF'    ),
    ('Heating'        , dp.VocabEncode()),
    ('HeatingQC'      , dp.VocabEncode()),
    ('CentralAir'     , dp.VocabEncode()),
    ('Electrical'     , dp.VocabEncode()),
    ('1stFlrSF'       ),
    ('2ndFlrSF'       ),
    ('LowQualFinSF'   ),
    ('GrLivArea'      ),
    ('BsmtFullBath'   ),
    ('BsmtHalfBath'   ),
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
    ('GarageCars'     ),
    ('GarageArea'     ),
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
    ('SalePrice'      ),
)
dp.transformAll(train_df, dp.Apply(float), except_columns = [])

train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['SalePrice'])
dp.save_df_to_csv(train_target_df, 'processed/train_target')
dp.save_df_to_csv(train_input_df, 'processed/train_input')

dp.analysis(train_df)
print(train_target_df.head(10))
print(train_input_df.head(10))