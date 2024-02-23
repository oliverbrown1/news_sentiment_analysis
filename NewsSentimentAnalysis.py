# dependencies - pandas and newsapi (install using pip install)
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import time

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

newsapi = NewsApiClient(api_key='032c11aeaff84f149dbc9f971ef0a389')


def get_news(company):

    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)

    to_date.strftime("%Y-%M-%D")
    from_date.strftime("%Y-%M-%D")

    data = newsapi.get_everything(q=company, from_param=from_date, to=to_date, sort_by="popularity", language="en",
                                  page_size=10)
    return data['articles']


def analyse_article(title, analyser):

    content = title
    print(analyser(content))


# This takes 2 seconds, might be a bit long for user so we can call this on startup of the backend server?
# creating and loading AI model with pretrained data (from transformers library), brought together by Hugging Face
# pipeline, which automatically does this for us.
def get_model():
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sentiment_analysis_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("finished loading model")
    return sentiment_analysis_model


analyser = get_model()
df = pd.DataFrame(get_news("microsoft"))
for index, row in df.iterrows():
    analyse_article(row['title'], analyser)


analyse_article("The really good company has risen in stock prices",analyser)
analyse_article("The really bad company has a fall in stock prices",analyser)
analyse_article("The company is alright",analyser)
