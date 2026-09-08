from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from news_signal_v2.models import (
    Article,
    ArticleExtractionError,
    NewsProviderError,
    NewsSearchResult,
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
        fallback_threshold: int = 5,
        client: Any | None = None,
    ) -> None:
        if fallback_threshold < 1:
            raise ValueError("fallback_threshold must be at least 1")
        if client is None:
            client = httpx.Client(timeout=20.0)
        self._api_key = api_key
        self._api_url = api_url
        self._domains = domains
        self._fallback_threshold = fallback_threshold
        self._client = client

    def fetch(
        self,
        search_terms: tuple[str, ...],
        ticker: str | None,
        lookback_days: int,
        cutoff_date: datetime | None = None,
    ) -> NewsSearchResult:
        to_date = cutoff_date or datetime.now(timezone.utc)
        if to_date.tzinfo is None:
            raise ValueError("cutoff_date must include a timezone")
        to_date = to_date.astimezone(timezone.utc)
        from_date = to_date - timedelta(days=lookback_days)
        quoted_terms = [f'"{term}"' for term in search_terms]
        query = " OR ".join(quoted_terms)
        if len(quoted_terms) > 1:
            query = f"({query})"
        del ticker

        params: dict[str, str | int] = {
            "q": query,
            "searchIn": "title",
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 100,
            "apiKey": self._api_key,
        }
        if not self._domains:
            return NewsSearchResult(tuple(self._request(params)), "all_domains", query)

        preferred_params = {**params, "domains": ",".join(self._domains)}
        preferred = self._request(preferred_params)
        if len(preferred) >= self._fallback_threshold:
            return NewsSearchResult(tuple(preferred), "configured_domains", query)

        # if not enough relevant articles scraped, will use unrestricted domains fallback
        unrestricted = self._request(params)
        return NewsSearchResult(
            tuple(preferred + unrestricted), "all_domains_fallback", query
        )

    def _request(self, params: dict[str, str | int]) -> list[Article]:
        try:
            response = self._client.get(self._api_url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = _error_message(exc.response)
            raise NewsProviderError(
                f"NewsAPI request failed with HTTP {exc.response.status_code}: {message}"
            ) from exc
        except httpx.RequestError as exc:
            raise NewsProviderError(
                f"NewsAPI request failed before receiving a response: {type(exc).__name__}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsProviderError("NewsAPI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise NewsProviderError("NewsAPI response must be an object")
        if payload.get("status") != "ok":
            raise NewsProviderError(
                f"NewsAPI returned an error: "
                f"{payload.get('message') or 'unknown provider error'}"
            )
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
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                else:
                    parsed_date = parsed_date.astimezone(timezone.utc)
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


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.reason_phrase or "unknown provider error"
    if isinstance(payload, dict) and payload.get("message"):
        return str(payload["message"])
    return response.reason_phrase or "unknown provider error"


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
