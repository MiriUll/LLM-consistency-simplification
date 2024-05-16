from transformers import pipeline
import pandas as pd
import torch
from tqdm import tqdm

print(torch.cuda.is_available())

from prediction_utils import pred_with_pipeline


print('*Loading Pipeline models')
pipeline_params = {'device': "cuda:0", 'max_length': 512, 'truncation': True, 'batch_size': 64}
# Load English emotion model
emotion_classifier_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                   **pipeline_params)
# Load topic model
topic_pipe = pipeline("text-classification", model="JiaqiLee/bert-agnews", **pipeline_params)
# Load sentiment model
sentiment_model_pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment",
                                **pipeline_params)
# Fake news
fake_pipe = pipeline("text-classification", model="hamzab/roberta-fake-news-classification", **pipeline_params)

data_path = ""

prediction_tasks = {
    'emotion': emotion_classifier_pipe,
    'topic': topic_pipe,
    #'sentiment': sentiment_model_pipe,
    'fake_news': fake_pipe,
    'sentiment_2':sentiment_model_pipe
}

def predict_and_store(column_name, identifier, filename):
    outname = filename if '_pred' in filename else filename.replace('.', '_pred.')
    df = pd.read_csv(data_path + filename)
    print('*', column_name)
    samples = list(df[column_name].dropna().unique()) # only predict the normal samples once
    for task_name, task_pipe in prediction_tasks.items():
        print('**', task_name)
        if task_name + identifier in df:
            print("\tdone")
            continue
        pipe_pred = pred_with_pipeline(task_pipe, [f"<title> <content> {s} <end>" for s in samples]) \
            if task_name == "fake_news" else pred_with_pipeline(task_pipe, samples)
        task_preds = {s: p for s, p in zip(samples, pipe_pred)}
        df[task_name + identifier] = df[column_name].apply(lambda x: task_preds.get(x))
        df.to_csv(data_path + outname, index=False)



#predict_and_store('simple_phrase', '_simple', 'data/newsela_sent_aligned_merged_pred.csv')
#predict_and_store('normal_phrase', '_normal', 'data/newsela_sent_aligned_merged_pred.csv')
#predict_and_store('simple_phrase', '_simple', 'data/newsela_sent_aligned_V0_pred.csv')
#predict_and_store('normal_phrase', '_normal', 'data/newsela_sent_aligned_V0_pred.csv')
#predict_and_store('simple_phrase', '_simple', 'data/newsela_sent_aligned_V0_pred_GPT3.5.csv')
#predict_and_store('normal_phrase', '_normal', 'data/newsela_sent_aligned_V0_pred_GPT3.5.csv')
predict_and_store('simple_phrase_ne', '_simple', 'data/newsela_sent_aligned_entities_masked_pred.csv')
predict_and_store('normal_phrase_ne', '_normal', 'data/newsela_sent_aligned_entities_masked_pred.csv')
