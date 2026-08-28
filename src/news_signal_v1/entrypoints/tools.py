from __future__ import annotations

from news_signal_v1.application import build_pipeline
from news_signal_v1.config import Settings
from news_signal_v1.pipeline import NewsAnalysisPipeline


class NewsSignalTools:
    def __init__(
        self,
        pipeline: NewsAnalysisPipeline,
        default_lookback_days: int = 7,
    ) -> None:
        self._pipeline = pipeline
        self._default_lookback_days = default_lookback_days

    def analyse_company_news(
        self,
        company: str,
        limit: int = 5,
        lookback_days: int | None = None,
    ) -> dict[str, object]:
        """Retrieve and analyse recent financial news about a company."""
        days = lookback_days if lookback_days is not None else self._default_lookback_days
        return self._pipeline.analyse(company, limit, days).to_dict()


def build_tools(settings: Settings | None = None) -> NewsSignalTools:
    settings = settings or Settings.from_env()
    return NewsSignalTools(pipeline=build_pipeline(settings), default_lookback_days=settings.lookback_days)
