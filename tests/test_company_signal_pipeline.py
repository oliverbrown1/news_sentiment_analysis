from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from company_signals.models import Filing, PriceBar, SignalFailure
from company_signals.pipeline import CompanySignalPipeline
from news_signal_v2.models import (
    AnalysisResult,
    AnalysedArticle,
    Article,
    SentimentResult,
)


class FakeNewsAnalyser:
    def __init__(self, published_at: datetime) -> None:
        self.published_at = published_at
        self.cutoff_date: datetime | None = None

    def analyse(
        self,
        company: str,
        ticker: str | None = None,
        limit: int = 5,
        lookback_days: int = 7,
        cutoff_date: datetime | None = None,
    ) -> AnalysisResult:
        del lookback_days
        assert cutoff_date is not None
        self.cutoff_date = cutoff_date
        article = Article(
            "Example raises guidance",
            "Reuters",
            "https://example.com/article",
            self.published_at,
        )
        return AnalysisResult(
            company,
            ticker,
            (
                AnalysedArticle(
                    article,
                    "Example raised guidance.",
                    SentimentResult("positive", 0.9),
                ),
            ),
            articles_eligible=1,
            articles_attempted=1,
            analysis_limit=limit,
        )


class FakePriceProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []

    @property
    def source(self) -> str:
        return "fake-prices"

    def fetch(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        self.calls.append((ticker, start, end))
        return [
            PriceBar(start + timedelta(days=index), 100, 100, 100, 100 + index, 1_000)
            for index in range(22)
        ]


class FakeFilingProvider:
    def latest(
        self, ticker: str, cutoff_date: datetime, forms: tuple[str, ...]
    ) -> Filing:
        del ticker, forms
        return Filing(
            "10-Q",
            cutoff_date - timedelta(days=10),
            "0001",
            "https://sec.test/0001",
        )


def test_pipeline_collects_typed_signals_and_preserves_evidence() -> None:
    cutoff_date = datetime(2024, 2, 20, tzinfo=timezone.utc)
    news_analyser = FakeNewsAnalyser(cutoff_date - timedelta(hours=3))
    prices = FakePriceProvider()
    pipeline = CompanySignalPipeline(news_analyser, prices, FakeFilingProvider())

    result = pipeline.collect("Example Ltd", "exm", cutoff_date)

    assert result.ticker == "EXM"
    assert news_analyser.cutoff_date == cutoff_date
    assert [call[0] for call in prices.calls] == ["EXM", "SPY"]
    assert prices.calls[0][2] == cutoff_date.date()
    assert {signal.group for signal in result.signals} == {
        "news",
        "price_momentum",
        "market_activity",
        "relative_performance",
        "fundamental",
    }
    assert {item.kind for item in result.evidence} == {"news", "filing"}
    assert result.news_stats.eligible == 1
    assert result.news_stats.attempted == 1
    assert result.news_stats.analysed == 1
    assert result.news_stats.limit == 20
    assert result.failures == ()


def test_bundle_output_is_compact_unless_verbose() -> None:
    cutoff_date = datetime(2024, 2, 20, tzinfo=timezone.utc)
    pipeline = CompanySignalPipeline(
        FakeNewsAnalyser(cutoff_date - timedelta(hours=3)),
        FakePriceProvider(),
        FakeFilingProvider(),
    )
    result = pipeline.collect("Example Ltd", "EXM", cutoff_date)
    result = replace(
        result,
        failures=(
            SignalFailure("news", "extraction", "blocked", "https://example.com/1"),
            SignalFailure("news", "extraction", "paywall", "https://example.com/2"),
        ),
    )

    compact = result.to_dict()
    verbose = result.to_dict(verbose=True)

    assert compact["news_stats"] == {
        "eligible": 1,
        "attempted": 1,
        "analysed": 1,
        "limit": 20,
    }
    assert "excerpt" not in compact["evidence"][0]
    assert compact["failures"] == []
    assert verbose["evidence"][0]["excerpt"] == "Example raised guidance."
    assert verbose["failures"][0]["reason"] == "blocked"

    fetch_failure = replace(
        result,
        failures=(SignalFailure("news", "fetch", "NewsAPI unavailable"),),
    )
    assert fetch_failure.to_dict()["failures"] == [
        {"source": "news", "stage": "fetch", "count": 1}
    ]
