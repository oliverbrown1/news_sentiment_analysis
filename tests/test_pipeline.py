from dataclasses import dataclass, field

import pytest

from news_signal_v1.adapters import ArticleExtractionError
from news_signal_v1.models import Article, SentimentResult
from news_signal_v1.pipeline import NewsAnalysisPipeline


@dataclass
class FakeNewsProvider:
    articles: list[Article]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def fetch(self, company: str, lookback_days: int) -> list[Article]:
        self.calls.append((company, lookback_days))
        return self.articles


class FakeExtractor:
    def __init__(self, summaries: dict[str, str]) -> None:
        self.summaries = summaries

    def extract(self, url: str) -> str:
        if url not in self.summaries:
            raise ArticleExtractionError(f"could not extract article: {url}")
        return self.summaries[url]


class FakeClassifier:
    def classify(self, title: str, content: str) -> SentimentResult:
        return SentimentResult("positive", 0.91)


def article(url: str, title: str = "Profits rise") -> Article:
    return Article(title=title, source_name="Reuters", url=url)


def test_pipeline_analyses_articles() -> None:
    provider = FakeNewsProvider([article("https://example.com/one")])
    pipeline = NewsAnalysisPipeline(
        provider,
        FakeExtractor({"https://example.com/one": "Revenue increased."}),
        FakeClassifier(),
    )

    result = pipeline.analyse("Example Ltd", lookback_days=14)

    assert provider.calls == [("Example Ltd", 14)]
    assert result.articles[0].sentiment.label == "positive"
    assert result.articles[0].sentiment.confidence == 0.91


def test_pipeline_skips_duplicates_and_records_extraction_failures() -> None:
    failed = article("https://example.com/failed")
    success = article("https://example.com/success")
    provider = FakeNewsProvider([failed, failed, success])
    pipeline = NewsAnalysisPipeline(
        provider,
        FakeExtractor({success.url: "Operating profit increased."}),
        FakeClassifier(),
    )

    result = pipeline.analyse("Example Ltd")

    assert len(result.articles) == 1
    assert len(result.failures) == 1
    assert result.failures[0].url == failed.url


def test_pipeline_respects_success_limit() -> None:
    articles = [article(f"https://example.com/{index}") for index in range(3)]
    summaries = {item.url: "Summary" for item in articles}
    pipeline = NewsAnalysisPipeline(
        FakeNewsProvider(articles), FakeExtractor(summaries), FakeClassifier()
    )

    result = pipeline.analyse("Example Ltd", limit=2)

    assert len(result.articles) == 2


@pytest.mark.parametrize(
    "company, limit, lookback_days",
    [("", 1, 7), ("Example Ltd", 0, 7), ("Example Ltd", 1, 0)],
)
def test_pipeline_validates_inputs(company: str, limit: int, lookback_days: int) -> None:
    pipeline = NewsAnalysisPipeline(FakeNewsProvider([]), FakeExtractor({}), FakeClassifier())

    with pytest.raises(ValueError):
        pipeline.analyse(company, limit=limit, lookback_days=lookback_days)
