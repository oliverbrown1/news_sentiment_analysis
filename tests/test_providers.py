from news_signal_v1.adapters import NewsApiProvider


class FakeNewsApiClient:
    def __init__(self) -> None:
        self.arguments = {}

    def get_everything(self, **kwargs):
        self.arguments = kwargs
        return {
            "articles": [
                {
                    "title": "Company raises guidance",
                    "source": {"name": "Reuters"},
                    "author": "Reporter",
                    "url": "https://example.com/article",
                    "publishedAt": "2026-08-23T09:00:00Z",
                }
            ]
        }


def test_news_api_provider_maps_response() -> None:
    client = FakeNewsApiClient()
    provider = NewsApiProvider("unused", client=client)

    articles = provider.fetch("Example Ltd", 7)

    assert client.arguments["q"] == "Example Ltd"
    assert client.arguments["language"] == "en"
    assert articles[0].source_name == "Reuters"
    assert articles[0].published_at is not None
    assert articles[0].published_at.isoformat() == "2026-08-23T09:00:00+00:00"
