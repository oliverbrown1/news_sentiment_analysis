from __future__ import annotations

import re
from html import unescape
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from news_signal_v2.models import (
    AnalysisFailure,
    AnalysisResult,
    AnalysedArticle,
    Article,
    ArticleExtractionError,
    EvidenceSelectionError,
    NewsSearchResult,
    SentimentClassificationError,
    SentimentResult,
)

TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class NewsProvider(Protocol):
    def fetch(
        self,
        search_terms: tuple[str, ...],
        ticker: str | None,
        lookback_days: int,
        cutoff_date: datetime | None = None,
    ) -> NewsSearchResult: ...


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
        self,
        company: str,
        ticker: str | None,
        title: str,
        content: str,
        aliases: tuple[str, ...] = (),
    ) -> str:
        sentences = _sentences(f"{title}. {content}")
        selected_indexes: set[int] = set()
        targets = (company, *aliases)
        for index, sentence in enumerate(sentences):
            company_matches = any(
                target.casefold() in sentence.casefold() for target in targets
            )
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
        cutoff_date: datetime | None = None,
        search_terms: tuple[str, ...] | None = None,
    ) -> AnalysisResult:
        company = company.strip()
        ticker = ticker.strip().upper() if ticker and ticker.strip() else None
        if not company:
            raise ValueError("company cannot be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if cutoff_date is not None and cutoff_date.tzinfo is None:
            raise ValueError("cutoff_date must include a timezone")
        terms = _normalise_search_terms(search_terms or (company,))

        search = self._news_provider.fetch(
            terms, ticker, lookback_days, cutoff_date
        )
        discovered = list(search.articles)
        # handles duplicate articles as well by normalising URLs
        articles, duplicates_removed = _deduplicate(discovered)
        # tracks specific failures as well
        analysed: list[AnalysedArticle] = []
        failures: list[AnalysisFailure] = []
        available_articles: list[Article] = []
        articles_attempted = 0
        articles_relevant = 0

        for article in articles:
            if cutoff_date is not None and not _available_by(article, cutoff_date):
                failures.append(
                    AnalysisFailure(
                        article.url,
                        "availability",
                        "article has no timestamp or was published after cutoff_date",
                    )
                )
                continue
            available_articles.append(article)

        for article in available_articles:
            articles_attempted += 1
            try:
                # extract URLs, do not summarise content with NLTK
                content = self._article_extractor.extract(article.url)
            except ArticleExtractionError as exc:
                failures.append(AnalysisFailure(article.url, "extraction", str(exc)))
                continue

            try:
                # select sentences with specific financial keywords that are relevant
                evidence = self._evidence_selector.select(
                    company, ticker, article.title, content, terms
                )
            except EvidenceSelectionError as exc:
                failures.append(AnalysisFailure(article.url, "relevance", str(exc)))
                continue
            articles_relevant += 1
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
            articles_retrieved=len(available_articles),
            articles_attempted=articles_attempted,
            articles_relevant=articles_relevant,
            analysis_limit=limit,
            lookback_days=lookback_days,
            search_strategy=search.strategy,
            search_query=search.query,
            search_terms=terms,
        )


def _available_by(article: Article, cutoff_date: datetime) -> bool:
    if article.published_at is None:
        return False
    published_at = article.published_at
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    return published_at <= cutoff_date


def _normalise_search_terms(search_terms: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in search_terms:
        term = value.strip()
        if not 2 <= len(term) <= 80:
            raise ValueError("each search term must contain between 2 and 80 characters")
        if '"' in term:
            raise ValueError("search terms cannot contain quotes")
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)
    if not terms:
        raise ValueError("at least one search term is required")
    if len(terms) > 5:
        raise ValueError("at most five search terms may be supplied")
    return tuple(terms)


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
