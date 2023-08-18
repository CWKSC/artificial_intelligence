import pandas as pd
import data_processing as dp
dp.init(__file__)

test_df = dp.read_csv("data/test")

test_df['GroupId'] = test_df['PassengerId'].apply(lambda x: x.split('_')[0])

groups = test_df['PassengerId'].apply(lambda x: x.split('_')[0]).value_counts().to_dict()
test_df['GroupSize'] = test_df['PassengerId'].apply(lambda x: groups[x.split('_')[0]])

test_df[['Deck', 'CabinNumber','CabinPosition']] = test_df['Cabin'].str.split('/', expand = True)
test_df.drop('Cabin', axis = 1, inplace = True)

test_df[['FirstName', 'LastName']] = test_df['Name'].str.split(' ', expand = True)
test_df.drop('Name', axis = 1, inplace = True)

age_mean = test_df['Age'].mean()

dp.transform(
    test_df,
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
    ('Deck'          , dp.VocabEncode()),
    ('CabinNumber'  , dp.VocabEncode()),
    ('CabinPosition', dp.VocabEncode()),
    ('FirstName'     , dp.VocabEncode()),
    ('LastName'      , dp.VocabEncode())
)
dp.transformAll(test_df, ['PassengerId'], dp.Apply(float))
dp.save_df_to_csv(test_df, 'processed/test')
