from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from news_signal_v2.models import Article, EvidenceSelectionError, SentimentResult
from news_signal_v2.pipeline import NewsAnalysisPipeline, TargetEvidenceSelector


@dataclass
class FakeProvider:
    articles: list[Article]

    def fetch(
        self,
        company: str,
        ticker: str | None,
        lookback_days: int,
        cutoff_date: datetime | None = None,
    ) -> list[Article]:
        return self.articles


class FakeExtractor:
    def extract(self, url: str) -> str:
        return (
            "The wider market was quiet. Example Ltd reported quarterly revenue "
            "growth and raised full-year guidance. Its shares gained 4%."
        )


class FakeClassifier:
    def classify(self, target: str, title: str, content: str) -> SentimentResult:
        return SentimentResult("positive", 0.92)


def test_v2_pipeline_deduplicates_and_classifies_target_evidence() -> None:
    provider = FakeProvider(
        [
            Article("Results", "Reuters", "https://example.com/results?utm_source=x"),
            Article("Results", "Reuters", "https://example.com/results#top"),
        ]
    )
    pipeline = NewsAnalysisPipeline(
        provider,
        FakeExtractor(),
        TargetEvidenceSelector(),
        FakeClassifier(),
    )

    result = pipeline.analyse("Example Ltd", ticker="EXM")

    assert result.duplicates_removed == 1
    assert result.articles_eligible == 1
    assert result.articles_attempted == 1
    assert result.analysis_limit == 5
    assert len(result.articles) == 1
    assert "Example Ltd reported" in result.articles[0].evidence
    assert result.articles[0].sentiment.label == "positive"


def test_ticker_selection_uses_word_boundaries() -> None:
    selector = TargetEvidenceSelector()

    evidence = selector.select("Unmentioned Corp", "AI", "", "They said revenue fell. AI gained.")

    assert evidence == "AI gained."


def test_evidence_selector_rejects_text_without_target() -> None:
    selector = TargetEvidenceSelector()

    with pytest.raises(EvidenceSelectionError, match="does not mention"):
        selector.select("Example Ltd", "EXM", "Market news", "Another firm gained.")


def test_v2_pipeline_rejects_articles_unavailable_at_prediction_time() -> None:
    provider = FakeProvider(
        [
            Article(
                "Future results",
                "Reuters",
                "https://example.com/future",
                datetime(2024, 1, 3, tzinfo=timezone.utc),
            )
        ]
    )
    pipeline = NewsAnalysisPipeline(
        provider,
        FakeExtractor(),
        TargetEvidenceSelector(),
        FakeClassifier(),
    )

    result = pipeline.analyse(
        "Example Ltd", cutoff_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
    )

    assert result.articles == ()
    assert result.articles_eligible == 0
    assert result.articles_attempted == 0
    assert result.failures[0].stage == "availability"


def test_v2_pipeline_reports_eligible_articles_beyond_analysis_limit() -> None:
    provider = FakeProvider(
        [
            Article("First result", "Reuters", "https://example.com/first"),
            Article("Second result", "Reuters", "https://example.com/second"),
        ]
    )
    pipeline = NewsAnalysisPipeline(
        provider,
        FakeExtractor(),
        TargetEvidenceSelector(),
        FakeClassifier(),
    )

    result = pipeline.analyse("Example Ltd", limit=1)

    assert result.articles_eligible == 2
    assert result.articles_attempted == 1
    assert result.analysis_limit == 1
    assert len(result.articles) == 1


@pytest.mark.parametrize(
    "company, limit, lookback_days",
    [("", 1, 7), ("Example Ltd", 0, 7), ("Example Ltd", 1, 0)],
)
def test_v2_pipeline_validates_inputs(
    company: str, limit: int, lookback_days: int
) -> None:
    pipeline = NewsAnalysisPipeline(
        FakeProvider([]),
        FakeExtractor(),
        TargetEvidenceSelector(),
        FakeClassifier(),
    )

    with pytest.raises(ValueError):
        pipeline.analyse(company, limit=limit, lookback_days=lookback_days)
