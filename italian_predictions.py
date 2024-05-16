from prediction_utils import LanguagePredictions
from transformers import pipeline

topic_pipe = pipeline("text2text-generation", model="mpapucci/it5-topic-classification-tag-it")
hate_pipe = pipeline("text-classification", model="IMSyPP/hate_speech_it")#"nickprock/setfit-italian-hate-speech")
emotion_pipe = pipeline("text-classification", model='MilaNLProc/feel-it-italian-emotion',top_k=1)
emotion_xlm_pipe = pipeline("text-classification", model='MilaNLProc/xlm-emo-t')
sentiment_pipe = pipeline("text-classification", model='neuraly/bert-base-italian-cased-sentiment')
sentiment_feel_pipe = pipeline("text-classification", model='MilaNLProc/feel-it-italian-sentiment')


prediction_tasks = {
    'topic': (topic_pipe, "Classifica Argomento: "),
    'hate_speech': (hate_pipe, ),
    'emotion': (emotion_pipe, ),
    'emotion_xlm': (emotion_xlm_pipe, ),
    'sentiment': (sentiment_pipe, ),
    'sentiment_feel': (sentiment_feel_pipe, )
}

lp = LanguagePredictions('data/', prediction_tasks)

lp.predict_and_store('normal_phrase', '_normal', 'admin_it_aligned_pred.csv')
lp.predict_and_store('simple_phrase', '_simple', 'admin_it_aligned_pred.csv')
lp.predict_and_store('normal_phrase', '_normal', 'simpitiki_pred.csv')
lp.predict_and_store('simple_phrase', '_simple', 'simpitiki_pred.csv')
lp.predict_and_store('normal_phrase', '_normal', 'corpus_simp_it_pred.csv')
lp.predict_and_store('simple_phrase', '_simple', 'corpus_simp_it_pred.csv')