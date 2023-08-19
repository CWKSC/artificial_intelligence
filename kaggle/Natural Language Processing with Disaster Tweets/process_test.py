from transformers import pipeline
import data_processing as dp
dp.init(__file__)

test_df = dp.read_csv('data/test')

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

dp.transform(
    test_df,
    ('id'        ),
    ('keyword'   , dp.VocabEncode()),
    ('location'  , dp.VocabEncode()),
    ('text'      , dp.Apply(lambda text: pipe(text)[0]['label']), dp.VocabEncode()),
)
dp.transformAll(test_df, dp.Apply(float), except_columns = ['id'])

test_target_df, test_input_df = dp.spilt_df(test_df, columns = ['id'])
dp.save_df_to_csv(test_target_df, 'processed/test_target')
dp.save_df_to_csv(test_input_df, 'processed/test_input')

dp.analysis(test_df)
print(test_target_df.head(10))
print(test_input_df.head(10))
