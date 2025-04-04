# dependencies - pandas, newspaper3k, newsapi, transformers and Pytorch (install using pip install)

# newspaper3k library used to fetch and summarise news content from a given URL, using a basic NLP for summarising
from newspaper import Article, Config, ArticleException
import nltk
import ssl
import config 

# NewsAPI library used to fetch news articles and their basic information (excluding content) given a company name
from newsapi import NewsApiClient
# import newsapi
from datetime import datetime, timedelta
# stored in pandas data frame
import pandas as pd

# transformers library of pretrained NLP models, and necessary frameworks to allow the Sentiment Analysis model to be created automatically
# Also uses Pytorch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import time
import json

import requests

# To download nltk NLP model (on Macbook)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# if punkt not installed, then add it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# NewsAPI key
newsapi = NewsApiClient(api_key=config.API_KEY)


# fetch news using newsAPI key and given company
def get_news(company):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)

    to_date.strftime("%Y-%M-%D")
    from_date.strftime("%Y-%M-%D")

    # 20 financial news domains now searched from, news is now more relevant to company and related to finance
    financial_domains = ['bloomberg.com', 'reuters.com', 'wsj.com', 'cnbc.com', 'ft.com', 'forbes.com',
                         'marketwatch.com', 'businessinsider.com', 'fool.com', 'investopedia.com', 'finance.yahoo.com',
                         'economist.com', 'thestreet.com', 'nasdaq.com', 'morningstar.com', 'investing.com',
                         'seekingalpha.com', 'cnbctv18.com', 'moneycontrol.com']

    data = newsapi.get_everything(q=company, domains=','.join(financial_domains), from_param=from_date, to=to_date,
                                  sort_by="relevancy", language="en")

    sources_in_response = set(article['source']['name'] for article in data['articles'])
    print(f"Sources in response: {sources_in_response}")

    return data['articles']


# fetch news content of given url, and summarise using simple NLP model from nltk
def get_news_content(url):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    newspaper_config = Config()
    newspaper_config.browser_user_agent = user_agent
    try:
        article = Article(url, config=newspaper_config)
        article.download()
        article.parse()
        article.nlp()

        summary = article.summary
        return summary
    # any errors or consent barriers that have to be passed manually
    except ArticleException as e:
        print(f"error occured: {e}")
        return None


# concatentating the title and summary into one piece of text, and performing Sentiment Analysis on this using the given analyser
def analyse_article(title, content, analyser):
    text = title + " " + content
    label = analyser(text)[0]['label']
    print(label)
    return label


# sentiment_model = None


# creating and loading AI model with pretrained data (from transformers library), brought together by Hugging Face
# pipeline, which automatically does this for us.
def get_model():
    # global sentiment_model

    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_model
    # sentiment_analysis_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("finished loading Sentiment Analysis model")


# Main module - fetch news on given company and store in a pandas data frame for exportation
def run_news_module(company, max_entries):
    global sentiment_model

    if sentiment_model == None:
        sentiment_model = get_model()

    df = pd.DataFrame(get_news(company))
    news_dataframe = pd.DataFrame(columns=['title', 'source_name', 'author', 'url', 'publish_date', 'sentiment'])
    for index, row in df.iterrows():
        news_summary = get_news_content(row['url'])
        if news_summary:
            label = analyse_article(row['title'], news_summary, sentiment_model)
            print(label)
            list_row = [row['title'], row['source']['name'], row['author'], row['url'], row['publishedAt'], label]
            news_dataframe.loc[len(news_dataframe)] = list_row
            # news_dataframe = news_dataframe._append([{'title': row['title'], 'source_name': row['source']['name'], 'author': row['author'], 'url': row['url'], 'publish_date': row['publishedAt'], 'sentiment': label}], ignore_index=True)
        if news_dataframe.shape[0] >= max_entries:
            break

    return news_dataframe
    # json_str = news_dataframe.to_json(orient='records')
    # return json.loads(json_str)


# KPI metric for a company
# counts appearance of each label, displaying the one that appears the most
def overall_sentiment(sentiment_list):
    sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
    for val in sentiment_list:
        sentiment_count[val] += 1

    if sentiment_count["positive"] == sentiment_count["negative"]:
        return {"neutral": 100}

    max_sentiment_count = 0
    max_label = ""
    for label in sentiment_count:
        if sentiment_count[label] > max_sentiment_count:
            max_sentiment_count = sentiment_count[label]
            max_label = label

    return {max_label: max_sentiment_count}


def run_test_to_analyse_10_articles(company):
    time_taken = 0
    sentiment_model = get_model()

    df = pd.DataFrame(get_news(company))
    news_dataframe = pd.DataFrame(columns=['title', 'source_name', 'author', 'url', 'publish_date', 'sentiment'])
    for index, row in df.iterrows():
        news_summary = get_news_content(row['url'])
        if news_summary:
            start_time = time.time()
            label = analyse_article(row['title'], news_summary, sentiment_model)
            end_time = time.time()
            time_taken += (end_time - start_time)
            list_row = [row['title'], row['source']['name'], row['author'], row['url'], row['publishedAt'], label]
            news_dataframe.loc[len(news_dataframe)] = list_row
            # news_dataframe = news_dataframe._append([{'title': row['title'], 'source_name': row['source']['name'], 'author': row['author'], 'url': row['url'], 'publish_date': row['publishedAt'], 'sentiment': label}], ignore_index=True)
        if news_dataframe.shape[0] >= 10:
            break

    return time_taken


def run_test_to_run_news_module(company):
    start_time = time.time()
    run_news_module(company, 5)
    end_time = time.time()
    return end_time - start_time

def test_model_accuracy():

    match_count = 0
    total = 0
    filename = 'Sentences_50Agree.txt'
    with open(filename, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    sentences = [line.split('@')[0].strip() for line in lines]
    sentiments = [line.split('@')[1].strip() for line in lines]

    sentiment_model = get_model()

    for i, sentence in enumerate(sentences):
        calculated_sentiment = analyse_article("", sentence, sentiment_model)
        total += 1

        if calculated_sentiment == sentiments[i]:
            match_count += 1
        else:
            print(f"match not found: {i}:{calculated_sentiment} found but {sentiments[i]} should be calculated")

    return match_count / total

print(f"{test_model_accuracy()} accurate")