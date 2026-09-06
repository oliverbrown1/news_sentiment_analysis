from datetime import datetime, timezone

from news_signal_v2.adapters import NewsApiProvider, TrafilaturaArticleExtractor


def _payload(count: int, source: str = "Reuters") -> dict[str, object]:
    return {
        "status": "ok",
        "articles": [
            {
                "title": f"Company update {index}",
                "source": {"name": source},
                "url": f"https://example.com/article-{index}",
                "publishedAt": "2026-08-23T09:00:00Z",
            }
            for index in range(count)
        ],
    }


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._payload


class FakeHttpClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = iter(payloads)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: dict[str, object]) -> FakeResponse:
        del url
        self.calls.append(params)
        return FakeResponse(next(self._payloads))


def test_v2_news_provider_uses_alias_in_titles_without_bare_ticker() -> None:
    client = FakeHttpClient([_payload(1)])
    provider = NewsApiProvider(
        "test-key", api_url="https://newsapi.test/everything", client=client
    )

    result = provider.fetch(
        ("Example",), "EXM", 7, datetime(2026, 8, 30, tzinfo=timezone.utc)
    )

    assert client.calls[0]["q"] == '"Example"'
    assert client.calls[0]["searchIn"] == "title"
    assert client.calls[0]["to"] == "2026-08-30T00:00:00+00:00"
    assert "domains" not in client.calls[0]
    assert result.strategy == "all_domains"
    assert result.articles[0].source_name == "Reuters"


def test_v2_news_provider_retries_without_domains_when_results_are_sparse() -> None:
    client = FakeHttpClient([_payload(1), _payload(3, "Other")])
    provider = NewsApiProvider(
        "test-key",
        api_url="https://newsapi.test/everything",
        domains=("reuters.com",),
        fallback_threshold=5,
        client=client,
    )

    result = provider.fetch(("Example", "Example Products"), "EXM", 7)

    assert client.calls[0]["q"] == '("Example" OR "Example Products")'
    assert client.calls[0]["domains"] == "reuters.com"
    assert "domains" not in client.calls[1]
    assert result.strategy == "all_domains_fallback"
    assert len(result.articles) == 4


def test_v2_news_provider_keeps_domains_when_results_are_sufficient() -> None:
    client = FakeHttpClient([_payload(5)])
    provider = NewsApiProvider(
        "test-key",
        api_url="https://newsapi.test/everything",
        domains=("reuters.com",),
        client=client,
    )

    result = provider.fetch(("Example",), "EXM", 7)

    assert len(client.calls) == 1
    assert result.strategy == "configured_domains"


def test_trafilatura_extractor_returns_clean_text() -> None:
    extractor = TrafilaturaArticleExtractor(
        fetcher=lambda url: "html",
        extractor=lambda html, **kwargs: "  Revenue increased.  ",
    )

    assert extractor.extract("https://example.com/article") == "Revenue increased."
