from news_signal.config import Settings
from news_signal.extraction import NewspaperArticleExtractor
from news_signal.pipeline import NewsAnalysisPipeline
from news_signal.providers import NewsApiProvider
from news_signal.sentiment import HuggingFaceSentimentClassifier


def build_pipeline(settings: Settings) -> NewsAnalysisPipeline:
    return NewsAnalysisPipeline(
        news_provider=NewsApiProvider(settings.news_api_key),
        article_extractor=NewspaperArticleExtractor(),
        sentiment_classifier=HuggingFaceSentimentClassifier(settings.sentiment_model),
    )
