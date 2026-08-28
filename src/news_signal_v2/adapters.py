from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from news_signal_v2.models import (
    Article,
    ArticleExtractionError,
    NewsProviderError,
    SentimentClassificationError,
    SentimentLabel,
    SentimentResult,
)

LABELS: dict[str, SentimentLabel] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

class NewsApiProvider:
    def __init__(
        self,
        api_key: str,
        *,
        api_url: str,
        domains: tuple[str, ...] = (),
        client: Any | None = None,
    ) -> None:
        if client is None:
            client = httpx.Client(timeout=20.0)
        self._api_key = api_key
        self._api_url = api_url
        self._domains = domains
        self._client = client

    def fetch(
        self, company: str, ticker: str | None, lookback_days: int
    ) -> list[Article]:
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=lookback_days)
        query = f'"{company}"'
        if ticker:
            query = f'("{company}" OR "{ticker}")'

        params: dict[str, str | int] = {
            "q": query,
            "from": from_date.date().isoformat(),
            "to": to_date.date().isoformat(),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 100,
            "apiKey": self._api_key,
        }
        if self._domains:
            params["domains"] = ",".join(self._domains)

        try:
            response = self._client.get(self._api_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NewsProviderError("NewsAPI request failed") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsProviderError("NewsAPI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise NewsProviderError("NewsAPI response must be an object")
        if payload.get("status") != "ok":
            raise NewsProviderError(str(payload.get("message") or "NewsAPI returned an error"))
        articles = payload.get("articles")
        if not isinstance(articles, list):
            raise NewsProviderError("NewsAPI response did not contain an article list")
        if not all(isinstance(item, dict) for item in articles):
            raise NewsProviderError("NewsAPI returned an invalid article")
        return [self._to_article(item) for item in articles]

    @staticmethod
    def _to_article(item: dict[str, Any]) -> Article:
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            raise NewsProviderError("NewsAPI article is missing a URL")
        source = item.get("source")
        if not isinstance(source, dict):
            source = {}
        published_at = item.get("publishedAt")
        parsed_date = None
        if published_at:
            try:
                parsed_date = datetime.fromisoformat(
                    str(published_at).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise NewsProviderError("NewsAPI article has an invalid date") from exc
        author = item.get("author")
        description = item.get("description")
        return Article(
            title=str(item.get("title") or "Untitled"),
            source_name=str(source.get("name") or "Unknown"),
            url=url.strip(),
            published_at=parsed_date,
            author=str(author) if author else None,
            description=str(description) if description else None,
        )


class TrafilaturaArticleExtractor:
    def __init__(
        self,
        fetcher: Callable[[str], Any] | None = None,
        extractor: Callable[..., str | None] | None = None,
    ) -> None:
        if fetcher is None or extractor is None:
            from trafilatura import extract, fetch_url

            fetcher = fetcher or fetch_url
            extractor = extractor or extract
        self._fetcher = fetcher
        self._extractor = extractor

    def extract(self, url: str) -> str:
        downloaded = self._fetcher(url)
        if downloaded is None:
            raise ArticleExtractionError(f"could not download article: {url}")
        text = self._extractor(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if not text or not text.strip():
            raise ArticleExtractionError(f"article produced no text: {url}")
        return text.strip()


class ModernFinBertSentimentClassifier:
    def __init__(self, model_name: str, classifier: Any | None = None) -> None:
        self._model_name = model_name
        self._classifier = classifier

    def classify(self, target: str, title: str, content: str) -> SentimentResult:
        del target
        classifier = self._classifier or self._load_classifier()
        try:
            output = classifier(f"{title} {content}".strip(), truncation=True)[0]
            raw_label = str(output["label"]).lower()
            confidence = float(output["score"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise SentimentClassificationError(
                f"invalid sentiment result from {self._model_name}"
            ) from exc
        label = LABELS.get(raw_label)
        if label is None:
            raise SentimentClassificationError(
                f"unsupported sentiment label from {self._model_name}: {raw_label}"
            )
        try:
            return SentimentResult(label=label, confidence=confidence)
        except ValueError as exc:
            raise SentimentClassificationError(
                f"invalid sentiment result from {self._model_name}"
            ) from exc

    def load(self) -> None:
        self._load_classifier()

    @property
    def model_revision(self) -> str | None:
        if self._classifier is None:
            return None
        model = getattr(self._classifier, "model", None)
        config = getattr(model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        return str(revision) if revision else None

    def _load_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        from transformers import pipeline

        self._classifier = pipeline(
            "text-classification",
            model=self._model_name,
            tokenizer=self._model_name,
        )
        return self._classifier
