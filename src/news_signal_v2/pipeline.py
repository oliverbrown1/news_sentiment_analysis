from __future__ import annotations

import re
from html import unescape
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from news_signal_v2.models import (
    AnalysisFailure,
    AnalysisResult,
    AnalysedArticle,
    Article,
    ArticleExtractionError,
    EvidenceSelectionError,
    SentimentClassificationError,
    SentimentResult,
)

TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class NewsProvider(Protocol):
    def fetch(
        self, company: str, ticker: str | None, lookback_days: int
    ) -> list[Article]: ...


class ArticleExtractor(Protocol):
    def extract(self, url: str) -> str: ...


class SentimentClassifier(Protocol):
    def classify(
        self, target: str, title: str, content: str
    ) -> SentimentResult: ...


class TargetEvidenceSelector:
    def __init__(self, max_characters: int = 2_000) -> None:
        self._max_characters = max_characters

    def select(
        self, company: str, ticker: str | None, title: str, content: str
    ) -> str:
        sentences = _sentences(f"{title}. {content}")
        selected_indexes: set[int] = set()
        for index, sentence in enumerate(sentences):
            company_matches = company.casefold() in sentence.casefold()
            ticker_matches = bool(
                ticker
                and re.search(rf"\b{re.escape(ticker)}\b", sentence, re.IGNORECASE)
            )
            if company_matches or ticker_matches:
                selected_indexes.add(index)
                if index + 1 < len(sentences):
                    selected_indexes.add(index + 1)

        if not selected_indexes:
            raise EvidenceSelectionError(
                f"extracted text does not mention {company}"
                + (f" or {ticker}" if ticker else "")
            )

        selected = " ".join(sentences[index] for index in sorted(selected_indexes))
        return selected[: self._max_characters].strip()


class NewsAnalysisPipeline:
    def __init__(
        self,
        news_provider: NewsProvider,
        article_extractor: ArticleExtractor,
        evidence_selector: TargetEvidenceSelector,
        sentiment_classifier: SentimentClassifier,
    ) -> None:
        self._news_provider = news_provider
        self._article_extractor = article_extractor
        self._evidence_selector = evidence_selector
        self._sentiment_classifier = sentiment_classifier

    def analyse(
        self,
        company: str,
        ticker: str | None = None,
        limit: int = 5,
        lookback_days: int = 7,
    ) -> AnalysisResult:
        company = company.strip()
        ticker = ticker.strip().upper() if ticker and ticker.strip() else None
        if not company:
            raise ValueError("company cannot be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")

        discovered = self._news_provider.fetch(company, ticker, lookback_days)
        # handles duplicate articles as well by normalising URLs
        articles, duplicates_removed = _deduplicate(discovered)
        # tracks specific failures as well
        analysed: list[AnalysedArticle] = []
        failures: list[AnalysisFailure] = []

        for article in articles:
            try:
                # extract URLs, do not summarise content with NLTK
                content = self._article_extractor.extract(article.url)
            except ArticleExtractionError as exc:
                failures.append(AnalysisFailure(article.url, "extraction", str(exc)))
                continue

            try:
                # select sentences with specific financial keywords that are relevant
                evidence = self._evidence_selector.select(
                    company, ticker, article.title, content
                )
            except EvidenceSelectionError as exc:
                failures.append(AnalysisFailure(article.url, "relevance", str(exc)))
                continue
            try:
                # classify sentiment
                sentiment = self._sentiment_classifier.classify(
                    company, article.title, evidence
                )
            except SentimentClassificationError as exc:
                failures.append(AnalysisFailure(article.url, "sentiment", str(exc)))
                continue

            analysed.append(AnalysedArticle(article, evidence, sentiment))
            if len(analysed) >= limit:
                break

        return AnalysisResult(
            company=company,
            ticker=ticker,
            articles=tuple(analysed),
            failures=tuple(failures),
            duplicates_removed=duplicates_removed,
        )


def _deduplicate(articles: list[Article]) -> tuple[list[Article], int]:
    unique: list[Article] = []
    urls: set[str] = set()
    titles: set[str] = set()
    for article in articles:
        url = _canonical_url(article.url)
        title = re.sub(r"[^a-z0-9]+", " ", article.title.casefold()).strip()
        if url in urls or title in titles:
            continue
        urls.add(url)
        titles.add(title)
        unique.append(article)
    return unique, len(articles) - len(unique)


def _canonical_url(url: str) -> str:
    parts = urlsplit(unescape(url))
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_PARAMETERS
    ]
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/") or "/",
            urlencode(query),
            "",
        )
    )


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
