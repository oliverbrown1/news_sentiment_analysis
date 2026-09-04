from __future__ import annotations

import os
from datetime import date, datetime

from company_signals.application import build_pipeline
from company_signals.entrypoints.arguments import documented
from company_signals.pipeline import CompanySignalPipeline
from company_signals.providers import SecCompanyResolver


class CompanySignalTools:
    def __init__(
        self,
        pipeline: CompanySignalPipeline,
        company_resolver: SecCompanyResolver | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._company_resolver = company_resolver

    @documented(
        "Resolve a company name and optional ticker against the SEC company list.",
        ("company", "ticker"),
    )
    def find_companies(
        self, company: str, ticker: str | None = None
    ) -> list[dict[str, str]]:
        if self._company_resolver is None:
            raise RuntimeError("company resolver is not configured")
        return [
            {"company": match.company, "ticker": match.ticker}
            for match in self._company_resolver.find(company, ticker)
        ]

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
        return _json_safe(
            self._pipeline.collect(
                company=company,
                ticker=ticker,
                cutoff_date=cutoff_date,
                benchmark=benchmark,
                news_days=news_days,
                news_limit=news_limit,
                price_days=price_days,
            ).to_dict(verbose=verbose)
        )

    @documented(
        "Calculate point-in-time news signals and return supporting evidence.",
        ("company", "ticker", "cutoff_date", "news_days", "news_limit"),
    )
    def get_news_signals(
        self,
        company: str,
        ticker: str,
        cutoff_date: datetime,
        news_days: int = 7,
        news_limit: int = 20,
    ) -> dict[str, object]:
        return _json_safe(
            self._pipeline.get_news_signals(
                company,
                ticker,
                cutoff_date,
                news_days=news_days,
                news_limit=news_limit,
            ).to_dict()
        )

    @documented(
        "Calculate point-in-time price and market comparison signals.",
        ("ticker", "cutoff_date", "benchmark", "price_days"),
    )
    def get_market_signals(
        self,
        ticker: str,
        cutoff_date: datetime,
        benchmark: str = "SPY",
        price_days: int = 45,
    ) -> dict[str, object]:
        return _json_safe(
            self._pipeline.get_market_signals(
                ticker,
                cutoff_date,
                benchmark=benchmark,
                price_days=price_days,
            ).to_dict()
        )

    @documented(
        "Return the latest supported SEC filing metadata available at the cutoff.",
        ("ticker", "cutoff_date"),
    )
    def get_filing_metadata(
        self,
        ticker: str,
        cutoff_date: datetime,
    ) -> dict[str, object]:
        return _json_safe(
            self._pipeline.get_filing_metadata(ticker, cutoff_date).to_dict()
        )


def build_tools() -> CompanySignalTools:
    pipeline = build_pipeline()
    return CompanySignalTools(
        pipeline,
        SecCompanyResolver(os.environ.get("SEC_USER_AGENT", "")),
    )


def _json_safe(result: dict[str, object]) -> dict[str, object]:
    return {key: _json_value(value) for key, value in result.items()}


def _json_value(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
