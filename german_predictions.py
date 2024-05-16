import pandas as pd
from germansentiment import SentimentModel
from transformers import pipeline
from simpletransformers.classification import ClassificationModel


# Load German sentiment model
sentiment_german_model = SentimentModel()

# Load German toxicity model
toxicity_pipeline = pipeline('text-classification', model='ml6team/distilbert-base-german-cased-toxic-comments', device='cuda:0')

# Load German topic model
model_args= {
            "num_train_epochs": 15,
            "learning_rate": 1e-5,
            "max_seq_length": 512,
            "silent": True
            }
topic_german_model = ClassificationModel(
    "xlmroberta", "classla/xlm-roberta-base-multilingual-text-genre-classifier", use_cuda=True,
    args=model_args
)

data_path = "data/"
datasets = {
    'TextComplexityDE_aligned.csv': pd.read_csv(data_path + 'TextComplexityDE_aligned.csv'),
}

def compare_preds(normal_pred:list, simple_pred:list, task:str):
  assert len(normal_pred) == len(simple_pred)
  same_pred = [n==s for n, s in zip(normal_pred, simple_pred)]
  equal = same_pred.count(True)
  different = same_pred.count(False)
  print(f"**{task} same predictions: {equal}, different predictions: {different}, error rate: {100*different/len(same_pred)}%")

for dataset_name, dataset_df in datasets.items():
  print("\n\n*", dataset_name)
  # 1. Sentiment prediction
  normal_sent_1 = sentiment_german_model.predict_sentiment(list(dataset_df.normal_phrase))
  # normal_sent = [sentiment_german_model.predict_sentiment(n) for n in dataset_df.normal_phrase]
  dataset_df['sentiment_normal'] = normal_sent_1
  simple_sent_1 = sentiment_german_model.predict_sentiment(list(dataset_df.simple_phrase))
  # simple_sent = [sentiment_german_model.predict_sentiment(s) for s in dataset_df.simple_phrase]
  dataset_df['sentiment_simple'] = simple_sent_1
  compare_preds(normal_sent_1, simple_sent_1, 'Sentiment')

  dataset_df.to_csv(data_path + dataset_name.replace('.', '_preds.'), index=False)

  # 2. Toxicity classification
  normal_tox = toxicity_pipeline(list(dataset_df.normal_phrase))
  normal_tox = [tox['label'] for tox in normal_tox]
  dataset_df['toxicity_normal'] = normal_tox
  simple_tox = toxicity_pipeline(list(dataset_df.simple_phrase))
  simple_tox = [tox['label'] for tox in simple_tox]
  dataset_df['toxicity_simple'] = simple_tox
  compare_preds(normal_tox, simple_tox, 'Toxicity')

  dataset_df.to_csv(data_path + dataset_name.replace('.', '_preds.'), index=False)

  # 3. Topic prediction
  normal_topic, _ = topic_german_model.predict(list(dataset_df.normal_phrase))
  dataset_df['topic_normal'] = normal_topic
  simple_topic, _ = topic_german_model.predict(list(dataset_df.simple_phrase))
  dataset_df['topic_simple'] = simple_topic
  compare_preds(normal_topic, simple_topic, 'Topic')

  dataset_df.to_csv(data_path + dataset_name.replace('.', '_preds.'), index=False)