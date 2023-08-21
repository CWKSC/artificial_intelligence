import numpy as np
import pandas as pd
from emoji import demojize
import matplotlib.pyplot as plt
import os, re, random, datasets, evaluate
pd.set_option('display.max_colwidth', None)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
import data_processing as dp
dp.init(__file__)

train = dp.read_csv("data/train")
test_df = dp.read_csv('data/test')

duplicates = train[train.duplicated('text')]
problematic_duplicates = []

for i in range(duplicates.text.nunique()):
    duplicate_subset = train[train.text == duplicates.text.unique()[i]]
    if len(duplicate_subset) > 1 and duplicate_subset.target.nunique() == 2:
        problematic_duplicates.append(i)

target_list = [0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0]

for problematic_index in range(len(problematic_duplicates)): 
    train.target = np.where(train.text == duplicates.text.unique()[problematic_index], 
                            target_list[problematic_index], train.target)
    

def clean_tweets(text):
    
    text = text.lower()

    text = text.replace("ca n't", "can't")
    text = text.replace("ai n't", "ain't")

    text = demojize(text)
    text = re.sub(r"^:[a-z,\_]+:$", " EMOJI ", text)
    
    # Contractions
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'd", "you would", text)
    text = re.sub(r"could've", "could have", text)
    
    text = re.sub(r" p . m .", " p.m.", text)
    text = re.sub(r" p . m ", " p.m ", text)
    text = re.sub(r" a . m .", " a.m.", text)
    text = re.sub(r" a . m ", " a.m ", text)

    text = re.sub(r'\n', '', text)
    text = re.sub(r'@\S+', '@USER', text)
    text = re.sub(r'http\S+', 'HTTPURL', text)
    text = re.sub(r'www\S+', 'HTTPURL', text)

    punctuations = '@#!?+&*[]-%.:/();$€£=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, '')
    
    text = text.replace("n't", " n't ")
    text = text.replace("n 't", " n't ")
    
    text = text.replace("'m", " 'm ")
    text = text.replace("'re", " 're ")
    text = text.replace("'s", " 's ")
    text = text.replace("'ll", " 'll ")
    text = text.replace("'d", " 'd ")
    text = text.replace("'ve", " 've ")
    text = text.replace("\n", " ")
    
    text = text.replace(" p . m .", " p.m.")
    text = text.replace(" p . m ", " p.m ")
    text = text.replace(" a . m .", " a.m.")
    text = text.replace(" a . m ", " a.m ")
    
    token_list = text.split(' ')
    
    token_list = [re.sub('#', '', x) for x in token_list]
    token_list = [re.sub(r'@\S+', '@USER', x) for x in token_list]
    token_list = [re.sub(r'http\S+', 'HTTPURL', x) for x in token_list]
    token_list = [re.sub(r'www\S+', 'HTTPURL', x) for x in token_list]
    token_list = [demojize(x) if len(x) == 1 else x for x in token_list]
    
    return(" ".join(token_list))

train.location = train.location.replace(np.nan, '', regex = True)
test_df.location = test_df.location.replace(np.nan, '', regex = True)

train.text = train.text + ". " + train.location + "."
test_df.text = test_df.text + ". " + test_df.location + "."

train.text = train.text.apply(lambda x: clean_tweets(x))
test_df.text = test_df.text.apply(lambda x: clean_tweets(x))

train = train.groupby('target').sample(np.min(train.target.value_counts().to_list()), random_state = 1048597)
train_df, val_df = np.split(train.sample(frac = 1), [int(0.8 * len(train))])

train_df = train_df[['id', 'text', 'target']]
val_df = val_df[['id', 'text', 'target']]
test_df = test_df[['id', 'text']]


train_dict = datasets.Dataset.from_dict(train_df.to_dict(orient="list"))
val_dict = datasets.Dataset.from_dict(val_df.to_dict(orient="list"))
test_dict = datasets.Dataset.from_dict(test_df.to_dict(orient="list"))

tweets_ds = datasets.DatasetDict({"train": train_dict, "val": val_dict, "test": test_dict})

model_name = 'C:\\Develop\\AI\\Model\\bertweet-base\\'
 # 'C:\\Develop\\AI\\Model\\bertweet-large\\' # 'vinai/bertweet-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)


def tokenize_function(dataset):
    return(tokenizer(dataset['text'], truncation = True))

tokenized_data = tweets_ds.map(tokenize_function, batched = True)


tokenized_data['train'] = tokenized_data['train'].rename_column('target', 'labels')
tokenized_data['val'] = tokenized_data['val'].rename_column('target', 'labels')
tokenized_data.with_format('pt')

# 3 0.00005 0.005 -> 0.8247
# 1 0.0001 0 -> 0.82163
# 1 0.0004 0 -> 0.57033
# 1 0.004 0 -> 


training_args = TrainingArguments(
    model_name,  
    evaluation_strategy = 'epoch',
    num_train_epochs = 1,
    learning_rate = 0.004,
    weight_decay = 0,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    report_to = 'none',
    load_best_model_at_end = True,
    save_strategy = 'epoch'
)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions=predictions, references=labels)

early_stop = EarlyStoppingCallback(2, 0.01)

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_data["train"],
    eval_dataset = tokenized_data["val"],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
    callbacks = [early_stop]
)


trainer.train()


test_predictions = trainer.predict(tokenized_data["test"])
preds = np.argmax(test_predictions.predictions, axis=1)

submission = pd.DataFrame(list(zip(test_df.id, preds)), 
                          columns = ["id", "target"])
dp.save_df_to_csv(submission, "submission/BERTweet")


