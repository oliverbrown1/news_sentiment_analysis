from __future__ import annotations

from datetime import datetime

from news_signal_v2.application import build_pipeline
from news_signal_v2.config import Settings
from news_signal_v2.entrypoints.arguments import documented
from news_signal_v2.pipeline import NewsAnalysisPipeline


class NewsSignalTools:
    def __init__(
        self, pipeline: NewsAnalysisPipeline, default_lookback_days: int
    ) -> None:
        self._pipeline = pipeline
        self._default_lookback_days = default_lookback_days

    @documented(
        "Retrieve and analyse company-specific financial news.",
        ("company", "ticker", "limit", "lookback_days", "cutoff_date"),
    )
    def analyse_company_news(
        self,
        company: str,
        ticker: str | None = None,
        limit: int = 5,
        lookback_days: int | None = None,
        cutoff_date: datetime | None = None,
    ) -> dict[str, object]:
        days = lookback_days or self._default_lookback_days
        if cutoff_date is None:
            result = self._pipeline.analyse(
                company=company,
                ticker=ticker,
                limit=limit,
                lookback_days=days,
            )
        else:
            result = self._pipeline.analyse(
                company=company,
                ticker=ticker,
                limit=limit,
                lookback_days=days,
                cutoff_date=cutoff_date,
            )
        return result.to_dict()


def build_tools(settings: Settings | None = None) -> NewsSignalTools:
    settings = settings or Settings.from_env()
    return NewsSignalTools(build_pipeline(settings), settings.lookback_days)
