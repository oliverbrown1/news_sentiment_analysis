from news_signal.adapters import (
    HuggingFaceSentimentClassifier,
    NewsApiProvider,
    NewspaperArticleExtractor,
)
from news_signal.config import Settings
from news_signal.pipeline import NewsAnalysisPipeline


def build_pipeline(settings: Settings) -> NewsAnalysisPipeline:
    return NewsAnalysisPipeline(
        news_provider=NewsApiProvider(settings.news_api_key),
        article_extractor=NewspaperArticleExtractor(),
        sentiment_classifier=HuggingFaceSentimentClassifier(settings.sentiment_model),
    )
