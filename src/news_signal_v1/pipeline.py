from __future__ import annotations

from typing import Protocol

from news_signal_v1.adapters import ArticleExtractionError
from news_signal_v1.models import AnalysisFailure, AnalysisResult, AnalysedArticle
from news_signal_v1.models import Article, SentimentResult


class NewsProvider(Protocol):
    def fetch(self, company: str, lookback_days: int) -> list[Article]: ...


class ArticleExtractor(Protocol):
    def extract(self, url: str) -> str: ...


class SentimentClassifier(Protocol):
    def classify(self, title: str, content: str) -> SentimentResult: ...


class NewsAnalysisPipeline:
    def __init__(
        self,
        news_provider: NewsProvider,
        article_extractor: ArticleExtractor,
        sentiment_classifier: SentimentClassifier,
    ) -> None:
        self._news_provider = news_provider
        self._article_extractor = article_extractor
        self._sentiment_classifier = sentiment_classifier

    def analyse(self, company: str, limit: int = 5, lookback_days: int = 7) -> AnalysisResult:
        company = company.strip()
        if not company:
            raise ValueError("company cannot be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")

        # fetch articles using newspaper3k
        articles = self._news_provider.fetch(company, lookback_days)
        analysed: list[AnalysedArticle] = []
        failures: list[AnalysisFailure] = []
        seen_urls: set[str] = set()

        for article in articles:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)

            # extract news article content and summarise using NLTK
            try:
                summary = self._article_extractor.extract(article.url)
            except ArticleExtractionError as exc:
                failures.append(AnalysisFailure(article.url, str(exc)))
                continue
            # analyse sentiment
            sentiment = self._sentiment_classifier.classify(article.title, summary)
            analysed.append(AnalysedArticle(article, summary, sentiment))
            if len(analysed) >= limit:
                break

        return AnalysisResult(company, tuple(analysed), tuple(failures))
