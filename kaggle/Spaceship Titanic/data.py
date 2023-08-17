import pandas as pd
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv("data/train")
clean_cabin = train_df['Cabin'].str.split('/', expand = True)
clean_cabin.columns = ["Deck", "Cabin_Number", "Cabin_Position"]
train_df = pd.concat([train_df, clean_cabin], axis = 1)
train_df.drop('Cabin', axis = 1, inplace = True)

test_df = dp.read_csv("data/test")
clean_cabin = test_df['Cabin'].str.split('/', expand = True)
clean_cabin.columns = ["Deck", "Cabin_Number", "Cabin_Position"]
test_df = pd.concat([test_df, clean_cabin], axis = 1)
test_df.drop('Cabin', axis = 1, inplace = True)

dp.transform(
    train_df,
    ('PassengerId'   , dp.Drop()),
    ('HomePlanet'    , dp.VocabEncode()),
    ('CryoSleep'     , dp.FillNa(-1), dp.Replace({'True': 1, 'False': 0})),
    ('Destination'   , dp.VocabEncode()),
    ('Age'           , dp.FillNa(-1)),
    ('VIP'           , dp.FillNa(-1), dp.Replace({'True': 1, 'False': 0})),
    ('RoomService'   , dp.FillNa(-1)),
    ('FoodCourt'     , dp.FillNa(-1)),
    ('ShoppingMall'  , dp.FillNa(-1)),
    ('Spa'           , dp.FillNa(-1)),
    ('VRDeck'        , dp.FillNa(-1)),
    ('Name'          , dp.Drop()),
    ('Transported'   , dp.Replace({'True': 1, 'False': 0})),
    ('Deck'          , dp.VocabEncode()),
    ('Cabin_Number'  , dp.VocabEncode()),
    ('Cabin_Position', dp.VocabEncode()),
)
dp.transformAll(train_df, [], dp.Apply(float))
dp.save_df_to_csv(train_df, 'processed/train')


dp.transform(
    test_df,
    ('HomePlanet'    , dp.VocabEncode()),
    ('CryoSleep'     , dp.FillNa(-1), dp.Replace({'True': 1, 'False': 0})),
    ('Destination'   , dp.VocabEncode()),
    ('Age'           , dp.FillNa(-1)),
    ('VIP'           , dp.FillNa(-1), dp.Replace({'True': 1, 'False': 0})),
    ('RoomService'   , dp.FillNa(-1)),
    ('FoodCourt'     , dp.FillNa(-1)),
    ('ShoppingMall'  , dp.FillNa(-1)),
    ('Spa'           , dp.FillNa(-1)),
    ('VRDeck'        , dp.FillNa(-1)),
    ('Name'          , dp.Drop()),
    ('Deck'          , dp.VocabEncode()),
    ('Cabin_Number'  , dp.VocabEncode()),
    ('Cabin_Position', dp.VocabEncode()),
)
dp.transformAll(test_df, ['PassengerId'], dp.Apply(float))
dp.save_df_to_csv(test_df, 'processed/test')
