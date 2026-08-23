from __future__ import annotations

from news_signal.extraction import ArticleExtractionError
from news_signal.interfaces import ArticleExtractor, NewsProvider, SentimentClassifier
from news_signal.models import AnalysisFailure, AnalysisResult, AnalysedArticle


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

        articles = self._news_provider.fetch(company, lookback_days)
        analysed: list[AnalysedArticle] = []
        failures: list[AnalysisFailure] = []
        seen_urls: set[str] = set()

        for article in articles:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)

            try:
                summary = self._article_extractor.extract(article.url)
            except ArticleExtractionError as exc:
                failures.append(AnalysisFailure(article.url, str(exc)))
                continue

            sentiment = self._sentiment_classifier.classify(article.title, summary)
            analysed.append(AnalysedArticle(article, summary, sentiment))
            if len(analysed) >= limit:
                break

        return AnalysisResult(company, tuple(analysed), tuple(failures))
