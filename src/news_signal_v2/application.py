from news_signal_v2.adapters import (
    ModernFinBertSentimentClassifier,
    NewsApiProvider,
    TrafilaturaArticleExtractor,
)
from news_signal_v2.config import Settings
from news_signal_v2.pipeline import NewsAnalysisPipeline, TargetEvidenceSelector


def build_pipeline(settings: Settings) -> NewsAnalysisPipeline:
    return NewsAnalysisPipeline(
        news_provider=NewsApiProvider(
            settings.news_api_key,
            api_url=settings.news_api_url,
            domains=settings.news_domains,
        ),
        article_extractor=TrafilaturaArticleExtractor(),
        evidence_selector=TargetEvidenceSelector(),
        sentiment_classifier=ModernFinBertSentimentClassifier(settings.sentiment_model),
    )
