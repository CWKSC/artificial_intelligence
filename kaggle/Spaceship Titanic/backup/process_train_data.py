import pandas as pd
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv("data/train")

train_df['GroupId'] = train_df['PassengerId'].apply(lambda x: x.split('_')[0])

groups = train_df['PassengerId'].apply(lambda x: x.split('_')[0]).value_counts().to_dict()
train_df['GroupSize'] = train_df['PassengerId'].apply(lambda x: groups[x.split('_')[0]])
train_df.drop('PassengerId', axis = 1, inplace=True)

train_df[['Deck', 'CabinNumber','CabinPosition']] = train_df['Cabin'].str.split('/', expand = True)
train_df.drop('Cabin', axis = 1, inplace = True)

train_df[['FirstName', 'LastName']] = train_df['Name'].str.split(' ', expand = True)
train_df.drop('Name', axis = 1, inplace = True)

age_mean = train_df['Age'].mean()

dp.transform(
    train_df,
    ('HomePlanet'    , dp.VocabEncode()),
    ('CryoSleep'     , dp.FillNa('False'), dp.Replace({'True': 1, 'False': 0})),
    ('Destination'   , dp.VocabEncode()),
    ('Age'           , dp.FillNa(age_mean)),
    ('VIP'           , dp.FillNa('False'), dp.Replace({'True': 1, 'False': 0})),
    ('RoomService'   , dp.FillNa(-1)),
    ('FoodCourt'     , dp.FillNa(-1)),
    ('ShoppingMall'  , dp.FillNa(-1)),
    ('Spa'           , dp.FillNa(-1)),
    ('VRDeck'        , dp.FillNa(-1)),
    ('Transported'   , dp.Replace({'True': 1, 'False': 0})),
    ('Deck'          , dp.VocabEncode()),
    ('CabinNumber'   , dp.VocabEncode()),
    ('CabinPosition' , dp.VocabEncode()),
    ('FirstName'     , dp.VocabEncode()),
    ('LastName'      , dp.VocabEncode())
)
dp.transformAll(train_df, dp.Apply(float), except_columns = [])
dp.analysis(train_df)

target_df, input_df = dp.spilt_df(train_df, columns = ['Transported'])
dp.save_df_to_csv(target_df, 'processed/train_target')
dp.save_df_to_csv(input_df, 'processed/train_input')

