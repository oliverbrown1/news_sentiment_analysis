from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True, slots=True)
class Article:
    title: str
    source_name: str
    url: str
    published_at: datetime | None = None
    author: str | None = None


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
    summary: str
    sentiment: SentimentResult


@dataclass(frozen=True, slots=True)
class AnalysisFailure:
    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    company: str
    articles: tuple[AnalysedArticle, ...]
    failures: tuple[AnalysisFailure, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
