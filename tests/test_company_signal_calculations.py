from datetime import date, datetime, timedelta, timezone

import pytest

from company_signals.calculations import (
    calculate_market_signals,
    calculate_news_signals,
)
from company_signals.models import PriceBar
from news_signal_v2.models import (
    AnalysisResult,
    AnalysedArticle,
    Article,
    SentimentResult,
)


def test_news_signals_aggregate_sentiment_volume_disagreement_and_recency() -> None:
    cutoff_date = datetime(2024, 1, 10, 12, tzinfo=timezone.utc)
    result = AnalysisResult(
        company="Example Ltd",
        ticker="EXM",
        articles=(
            AnalysedArticle(
                Article(
                    "Strong results",
                    "Reuters",
                    "https://example.com/positive",
                    cutoff_date - timedelta(hours=2),
                ),
                "Revenue increased.",
                SentimentResult("positive", 0.9),
            ),
            AnalysedArticle(
                Article(
                    "Mixed outlook",
                    "Reuters",
                    "https://example.com/neutral",
                    cutoff_date - timedelta(hours=5),
                ),
                "Guidance was unchanged.",
                SentimentResult("neutral", 0.6),
            ),
        ),
        articles_eligible=2,
        articles_attempted=2,
        analysis_limit=20,
    )

    signals = {
        signal.name: signal
        for signal in calculate_news_signals(result, cutoff_date)
    }

    assert signals["articles_eligible"].value == 2
    assert signals["sentiment_score"].value == pytest.approx(0.6)
    assert signals["sentiment_disagreement"].value == 0.5
    assert signals["latest_article_age"].value == 2.0


def test_market_signals_use_only_bars_available_before_cutoff_date() -> None:
    first_day = date(2024, 1, 1)
    company = [
        _bar(first_day + timedelta(days=index), 100 + index, 1_000 + index)
        for index in range(22)
    ]
    benchmark = [_bar(first_day + timedelta(days=index), 200, 2_000) for index in range(22)]
    cutoff_date = datetime(2024, 1, 22, tzinfo=timezone.utc)

    signals = {
        signal.name: signal
        for signal in calculate_market_signals(
            company, benchmark, cutoff_date, "yfinance"
        )
    }

    assert signals["return_1d"].value == pytest.approx(120 / 119 - 1)
    assert signals["return_5d"].value == pytest.approx(120 / 115 - 1)
    assert signals["return_20d"].value == pytest.approx(0.2)
    assert signals["benchmark_relative_return_5d"].value == pytest.approx(120 / 115 - 1)
    assert signals["relative_volume_20d"].value == pytest.approx(1_020 / 1_009.5)


def _bar(day: date, close: float, volume: int) -> PriceBar:
    return PriceBar(day, close, close, close, close, volume)
