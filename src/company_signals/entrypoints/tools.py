from __future__ import annotations

from datetime import datetime

from company_signals.application import build_pipeline
from company_signals.entrypoints.arguments import documented
from company_signals.pipeline import CompanySignalPipeline


class CompanySignalTools:
    def __init__(self, pipeline: CompanySignalPipeline) -> None:
        self._pipeline = pipeline

    @documented(
        "Collect point-in-time news, market, and filing signals for a company.",
        (
            "company",
            "ticker",
            "cutoff_date",
            "benchmark",
            "news_days",
            "news_limit",
            "price_days",
            "verbose",
        ),
    )
    def collect_signals(
        self,
        company: str,
        ticker: str,
        cutoff_date: datetime,
        benchmark: str = "SPY",
        news_days: int = 7,
        news_limit: int = 20,
        price_days: int = 45,
        verbose: bool = False,
    ) -> dict[str, object]:
        return self._pipeline.collect(
            company=company,
            ticker=ticker,
            cutoff_date=cutoff_date,
            benchmark=benchmark,
            news_days=news_days,
            news_limit=news_limit,
            price_days=price_days,
        ).to_dict(verbose=verbose)


def build_tools() -> CompanySignalTools:
    return CompanySignalTools(build_pipeline())
