from __future__ import annotations

from datetime import datetime, timezone
from statistics import pstdev

from company_signals.models import Filing, PriceBar, Signal
from news_signal_v2.models import AnalysisResult

SENTIMENT_SCORES = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def calculate_news_signals(
    result: AnalysisResult, cutoff_date: datetime
) -> list[Signal]:
    articles = [
        item
        for item in result.articles
        if item.article.published_at is not None
        and _utc(item.article.published_at) <= cutoff_date
    ]
    observed_at = max(
        (_utc(item.article.published_at) for item in articles),
        default=cutoff_date,
    )
    signals = [
        Signal(
            "news",
            "articles_eligible",
            result.articles_eligible,
            observed_at,
            observed_at,
            "newsapi",
            "articles",
        )
    ]
    if not articles:
        return signals

    scores = [SENTIMENT_SCORES[item.sentiment.label] for item in articles]
    weights = [item.sentiment.confidence for item in articles]
    total_weight = sum(weights)
    sentiment = (
        sum(score * weight for score, weight in zip(scores, weights, strict=True))
        / total_weight
        if total_weight
        else 0.0
    )
    recency_hours = max(
        0.0, (cutoff_date - observed_at).total_seconds() / 3_600
    )
    signals.extend(
        [
            Signal(
                "news",
                "sentiment_score",
                sentiment,
                observed_at,
                observed_at,
                "news_signal_v2",
                "score",
            ),
            Signal(
                "news",
                "sentiment_disagreement",
                pstdev(scores),
                observed_at,
                observed_at,
                "news_signal_v2",
                "score",
            ),
            Signal(
                "news",
                "latest_article_age",
                recency_hours,
                observed_at,
                observed_at,
                "newsapi",
                "hours",
            ),
        ]
    )
    return signals


def calculate_market_signals(
    company_bars: list[PriceBar],
    benchmark_bars: list[PriceBar],
    cutoff_date: datetime,
    price_source: str,
) -> list[Signal]:
    company = _available_bars(company_bars, cutoff_date)
    benchmark = _available_bars(benchmark_bars, cutoff_date)
    if not company:
        return []

    latest = company[-1]
    observed_at = datetime.combine(latest.session_date, datetime.min.time(), timezone.utc)
    available_at = latest.available_at
    signals: list[Signal] = []
    for days in (1, 5, 20):
        if len(company) > days:
            value = company[-1].close / company[-days - 1].close - 1
            signals.append(
                Signal(
                    "price_momentum",
                    f"return_{days}d",
                    value,
                    observed_at,
                    available_at,
                    price_source,
                    "decimal",
                )
            )

    if len(company) >= 21:
        returns = [
            company[index].close / company[index - 1].close - 1
            for index in range(len(company) - 20, len(company))
        ]
        average_volume = sum(bar.volume for bar in company[-21:-1]) / 20
        relative_volume = latest.volume / average_volume if average_volume else 0.0
        signals.extend(
            [
                Signal(
                    "market_activity",
                    "volatility_20d",
                    pstdev(returns),
                    observed_at,
                    available_at,
                    price_source,
                    "decimal",
                ),
                Signal(
                    "market_activity",
                    "relative_volume_20d",
                    relative_volume,
                    observed_at,
                    available_at,
                    price_source,
                    "ratio",
                ),
            ]
        )

    if len(company) >= 6 and len(benchmark) >= 6:
        relative_return = (
            company[-1].close / company[-6].close
            - benchmark[-1].close / benchmark[-6].close
        )
        signals.append(
            Signal(
                "relative_performance",
                "benchmark_relative_return_5d",
                relative_return,
                observed_at,
                available_at,
                price_source,
                "decimal",
            )
        )
    return signals


def calculate_filing_signals(
    filing: Filing, cutoff_date: datetime
) -> list[Signal]:
    age_days = (cutoff_date - filing.accepted_at).total_seconds() / 86_400
    return [
        Signal(
            "fundamental",
            "latest_filing_form",
            filing.form,
            filing.accepted_at,
            filing.accepted_at,
            "sec",
        ),
        Signal(
            "fundamental",
            "latest_filing_age",
            age_days,
            filing.accepted_at,
            filing.accepted_at,
            "sec",
            "days",
        ),
    ]


def _available_bars(
    bars: list[PriceBar], cutoff_date: datetime
) -> list[PriceBar]:
    return sorted(
        (bar for bar in bars if bar.available_at <= cutoff_date),
        key=lambda bar: bar.session_date,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
