from typing import Protocol

from news_signal.models import Article, SentimentResult


class NewsProvider(Protocol):
    def fetch(self, company: str, lookback_days: int) -> list[Article]: ...


class ArticleExtractor(Protocol):
    def extract(self, url: str) -> str: ...


class SentimentClassifier(Protocol):
    def classify(self, title: str, content: str) -> SentimentResult: ...
