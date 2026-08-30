from datetime import date

import pytest

from eval.market_eval import MarketPrediction
from eval.sentiment_baseline import SentimentMarketBaseline, aggregate_sentiment
from news_signal_v2.models import SentimentResult


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def classify(self, target: str, title: str, content: str) -> SentimentResult:
        self.calls.append((target, title, content))
        return SentimentResult("positive", 0.9)


def test_sentiment_baseline_maps_headline_sentiment_to_market_direction() -> None:
    classifier = FakeClassifier()
    baseline = SentimentMarketBaseline(classifier)

    first = baseline.predict(date(2024, 1, 2), "AAA", "Revenue increased")
    second = baseline.predict(date(2024, 1, 3), "BBB", "Revenue increased")

    assert first == MarketPrediction("positive", 0.9)
    assert second == first
    assert classifier.calls == [("AAA", "", "Revenue increased")]


def test_aggregate_sentiment_uses_confidence_weighted_direction() -> None:
    prediction = aggregate_sentiment(
        [MarketPrediction("positive", 0.9), MarketPrediction("negative", 0.3)]
    )

    assert prediction.label == "positive"
    assert prediction.confidence == pytest.approx(0.5)


def test_aggregate_sentiment_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_sentiment([])
