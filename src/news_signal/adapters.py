from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from news_signal.models import Article, SentimentLabel, SentimentResult

FINANCIAL_DOMAINS = (
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "cnbc.com",
    "ft.com",
    "forbes.com",
    "marketwatch.com",
    "businessinsider.com",
    "fool.com",
    "investopedia.com",
    "finance.yahoo.com",
    "economist.com",
    "thestreet.com",
    "nasdaq.com",
    "morningstar.com",
    "investing.com",
    "seekingalpha.com",
    "cnbctv18.com",
    "moneycontrol.com",
)

LABELS: dict[str, SentimentLabel] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
}


class ArticleExtractionError(RuntimeError):
    pass


class NewsApiProvider:
    def __init__(self, api_key: str, client: Any | None = None) -> None:
        if client is None:
            from newsapi import NewsApiClient

            client = NewsApiClient(api_key=api_key)
        self._client = client

    def fetch(self, company: str, lookback_days: int) -> list[Article]:
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=lookback_days)
        response = self._client.get_everything(
            q=company,
            domains=",".join(FINANCIAL_DOMAINS),
            from_param=from_date.date().isoformat(),
            to=to_date.date().isoformat(),
            sort_by="relevancy",
            language="en",
        )
        return [self._to_article(item) for item in response.get("articles", [])]

    @staticmethod
    def _to_article(item: dict[str, Any]) -> Article:
        source = item.get("source") or {}
        published_at = item.get("publishedAt")
        parsed_date = None
        if published_at:
            parsed_date = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))

        author = item.get("author")
        return Article(
            title=str(item.get("title") or "Untitled"),
            source_name=str(source.get("name") or "Unknown"),
            url=str(item["url"]),
            published_at=parsed_date,
            author=str(author) if author else None,
        )


class NewspaperArticleExtractor:
    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )

    def extract(self, url: str) -> str:
        from newspaper import Article as NewspaperArticle
        from newspaper import ArticleException, Config

        config = Config()
        config.browser_user_agent = self._user_agent
        article = NewspaperArticle(url, config=config)
        try:
            article.download()
            article.parse()
            article.nlp()
        except (ArticleException, LookupError) as exc:
            raise ArticleExtractionError(f"could not extract article: {url}") from exc

        summary = article.summary.strip()
        if not summary:
            raise ArticleExtractionError(f"article produced an empty summary: {url}")
        return summary


class HuggingFaceSentimentClassifier:
    def __init__(self, model_name: str, classifier: Any | None = None) -> None:
        self._model_name = model_name
        self._classifier = classifier

    def classify(self, title: str, content: str) -> SentimentResult:
        classifier = self._classifier or self._load_classifier()
        output = classifier(f"{title} {content}".strip(), truncation=True)[0]
        raw_label = str(output["label"]).lower()
        label = LABELS.get(raw_label)
        if label is None:
            raise ValueError(f"unsupported sentiment label: {raw_label}")
        return SentimentResult(label=label, confidence=float(output["score"]))

    def _load_classifier(self) -> Any:
        from transformers import pipeline

        self._classifier = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
        )
        return self._classifier
