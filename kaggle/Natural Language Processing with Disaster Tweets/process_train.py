from transformers import pipeline
import data_processing as dp
dp.init(__file__)

train_df = dp.read_csv("data/train")

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

dp.transform(
    train_df,
    ('id'        , dp.Drop()),
    ('keyword'   , dp.VocabEncode()),
    ('location'  , dp.VocabEncode()),
    ('text'      , dp.Apply(lambda text: pipe(text)[0]['label']), dp.VocabEncode()),
    ('target'    ),
)
dp.transformAll(train_df, dp.Apply(float), except_columns = [])

train_target_df, train_input_df = dp.spilt_df(train_df, columns = ['target'])
dp.save_df_to_csv(train_target_df, 'processed/train_target')
dp.save_df_to_csv(train_input_df, 'processed/train_input')

dp.analysis(train_df)
print(train_target_df.head(10))
print(train_input_df.head(10))

