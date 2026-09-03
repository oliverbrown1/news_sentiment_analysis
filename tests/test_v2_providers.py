from datetime import datetime, timezone

from news_signal_v2.adapters import NewsApiProvider, TrafilaturaArticleExtractor


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return {
            "status": "ok",
            "articles": [
                {
                    "title": "Company raises guidance",
                    "source": {"name": "Reuters"},
                    "url": "https://example.com/article",
                    "publishedAt": "2026-08-23T09:00:00Z",
                }
            ],
        }


class FakeHttpClient:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def get(self, url: str, *, params: dict[str, object]) -> FakeResponse:
        self.params = params
        return FakeResponse()


def test_v2_news_provider_uses_company_and_ticker_without_domain_filter() -> None:
    client = FakeHttpClient()
    provider = NewsApiProvider(
        "test-key", api_url="https://newsapi.test/everything", client=client
    )

    articles = provider.fetch(
        "Example Ltd", "EXM", 7, datetime(2026, 8, 30, tzinfo=timezone.utc)
    )

    assert client.params["q"] == '("Example Ltd" OR "EXM")'
    assert client.params["to"] == "2026-08-30T00:00:00+00:00"
    assert "domains" not in client.params
    assert articles[0].source_name == "Reuters"


def test_trafilatura_extractor_returns_clean_text() -> None:
    extractor = TrafilaturaArticleExtractor(
        fetcher=lambda url: "html",
        extractor=lambda html, **kwargs: "  Revenue increased.  ",
    )

    assert extractor.extract("https://example.com/article") == "Revenue increased."
