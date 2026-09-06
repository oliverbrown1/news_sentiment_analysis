from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

SentimentLabel = Literal["positive", "neutral", "negative"]
SearchStrategy = Literal[
    "configured_domains",
    "all_domains_fallback",
    "all_domains",
]


class NewsProviderError(RuntimeError):
    pass


class ArticleExtractionError(RuntimeError):
    pass


class SentimentClassificationError(RuntimeError):
    pass


class EvidenceSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Article:
    title: str
    source_name: str
    url: str
    published_at: datetime | None = None
    author: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class NewsSearchResult:
    articles: tuple[Article, ...]
    strategy: SearchStrategy
    query: str


@dataclass(frozen=True, slots=True)
class SentimentResult:
    label: SentimentLabel
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AnalysedArticle:
    article: Article
    evidence: str
    sentiment: SentimentResult


@dataclass(frozen=True, slots=True)
class AnalysisFailure:
    url: str
    stage: Literal["availability", "extraction", "relevance", "sentiment"]
    reason: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    company: str
    ticker: str | None
    articles: tuple[AnalysedArticle, ...]
    failures: tuple[AnalysisFailure, ...] = ()
    duplicates_removed: int = 0
    articles_retrieved: int = 0
    articles_attempted: int = 0
    articles_relevant: int = 0
    analysis_limit: int = 0
    lookback_days: int = 0
    search_strategy: SearchStrategy = "all_domains"
    search_query: str = ""
    search_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
