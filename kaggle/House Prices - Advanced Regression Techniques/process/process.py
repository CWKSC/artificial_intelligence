
import data_processing as dp
import pandas as pd
import data_analysis as da
dp.init(__file__)

train_df = dp.read_csv('../data/train')
test_df = dp.read_csv('../data/test')

def process(df: pd.DataFrame) -> pd.DataFrame:
    dp.transform(
        df,
        ('MSSubClass'   ),
        ('MSZoning'     , dp.FillNaWithMode(), dp.VocabEncode()),
        ('LotFrontage'  , dp.FillNaWithMean()),
        ('LotArea'      ),
        ('Street'       , dp.VocabEncode()),
        ('Alley'        , dp.FillNaWithMode(), dp.VocabEncode()),
        ('LotShape'     , dp.VocabEncode()),
        ('LandContour'  , dp.VocabEncode()),
        ('Utilities'    , dp.FillNaWithMode(), dp.VocabEncode()),
        ('LotConfig'    , dp.VocabEncode()),
        ('LandSlope'    , dp.VocabEncode()),
        ('Neighborhood' , dp.VocabEncode()),
        ('Condition1'   , dp.VocabEncode()),
        ('Condition2'   , dp.VocabEncode()),
        ('BldgType'     , dp.VocabEncode()),
        ('HouseStyle'   , dp.VocabEncode()),
        ('OverallQual'  ),
        ('OverallCond'  ),
        ('YearBuilt'    ),
        ('YearRemodAdd' ),
        ('RoofStyle'    , dp.VocabEncode()),
        ('RoofMatl'     , dp.VocabEncode()),
        ('Exterior1st'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Exterior2nd'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('MasVnrType'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('MasVnrArea'   , dp.FillNaWithMean()),
        ('ExterQual'    , dp.VocabEncode()),
        ('ExterCond'    , dp.VocabEncode()),
        ('Foundation'   , dp.VocabEncode()),
        ('BsmtQual'     , dp.FillNaWithMode(), dp.VocabEncode()),
        ('BsmtCond'     , dp.FillNaWithMode(), dp.VocabEncode()),
        ('BsmtExposure' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('BsmtFinType1' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('BsmtFinSF1'   , dp.FillNaWithMean()),
        ('BsmtFinType2' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('BsmtFinSF2'   , dp.FillNaWithMean()),
        ('BsmtUnfSF'    , dp.FillNaWithMean()),
        ('TotalBsmtSF'  , dp.FillNaWithMean()),
        ('Heating'      , dp.VocabEncode()),
        ('HeatingQC'    , dp.VocabEncode()),
        ('CentralAir'   , dp.VocabEncode()),
        ('Electrical'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('1stFlrSF'     ),
        ('2ndFlrSF'     ),
        ('LowQualFinSF' ),
        ('GrLivArea'    ),
        ('BsmtFullBath' , dp.FillNaWithMean()),
        ('BsmtHalfBath' , dp.FillNaWithMean()),
        ('FullBath'     ),
        ('HalfBath'     ),
        ('BedroomAbvGr' ),
        ('KitchenAbvGr' ),
        ('KitchenQual'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('TotRmsAbvGrd' ),
        ('Functional'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Fireplaces'   ),
        ('FireplaceQu'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('GarageType'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('GarageYrBlt'  , dp.FillNaWithMean()),
        ('GarageFinish' , dp.FillNaWithMode(), dp.VocabEncode()),
        ('GarageCars'   , dp.FillNaWithMean()),
        ('GarageArea'   , dp.FillNaWithMean()),
        ('GarageQual'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('GarageCond'   , dp.FillNaWithMode(), dp.VocabEncode()),
        ('PavedDrive'   , dp.VocabEncode()),
        ('WoodDeckSF'   ),
        ('OpenPorchSF'  ),
        ('EnclosedPorch'),
        ('3SsnPorch'    ),
        ('ScreenPorch'  ),
        ('PoolArea'     ),
        ('PoolQC'       , dp.FillNaWithMode(), dp.VocabEncode()),
        ('Fence'        , dp.FillNaWithMode(), dp.VocabEncode()),
        ('MiscFeature'  , dp.FillNaWithMode(), dp.VocabEncode()),
        ('MiscVal'      ),
        ('MoSold'       ),
        ('YrSold'       ),
        ('SaleType'     , dp.FillNaWithMode(), dp.VocabEncode()),
        ('SaleCondition', dp.VocabEncode()),
    )
    da.analysis_df(df)
    dp.transformAll(df, dp.Apply(float), except_columns = ['Id'])
    return df

train_df = process(train_df)
test_df = process(test_df)

train_df.drop(columns=['Id'], inplace=True)
dp.save_df_to_csv(train_df, '../processed/train')
train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['SalePrice'])
dp.save_df_to_csv(train_target_df, '../processed/train_target')
dp.save_df_to_csv(train_input_df, '../processed/train_input')

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['Id'])
dp.save_df_to_csv(test_target_df, '../processed/test_target')
dp.save_df_to_csv(test_input_df, '../processed/test_input')
