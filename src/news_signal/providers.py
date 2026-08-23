from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from news_signal.models import Article

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
