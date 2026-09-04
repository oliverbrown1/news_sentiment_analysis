from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

from company_signals.calculations import (
    calculate_filing_signals,
    calculate_market_signals,
    calculate_news_signals,
)
from company_signals.models import (
    EvidenceReference,
    NewsStats,
    SignalBundle,
    SignalFailure,
    SignalProviderError,
    SignalResult,
)
from company_signals.providers import FilingProvider, PriceProvider
from news_signal_v2.models import AnalysisResult, NewsProviderError

DEFAULT_FILING_FORMS = ("10-K", "10-Q", "8-K", "20-F", "6-K")


class NewsAnalyser(Protocol):
    def analyse(
        self,
        company: str,
        ticker: str | None = None,
        limit: int = 5,
        lookback_days: int = 7,
        cutoff_date: datetime | None = None,
    ) -> AnalysisResult: ...


class CompanySignalPipeline:
    def __init__(
        self,
        news_analyser: NewsAnalyser,
        price_provider: PriceProvider,
        filing_provider: FilingProvider,
    ) -> None:
        self._news_analyser = news_analyser
        self._price_provider = price_provider
        self._filing_provider = filing_provider

    def collect(
        self,
        company: str,
        ticker: str,
        cutoff_date: datetime,
        *,
        benchmark: str = "SPY",
        news_days: int = 7,
        news_limit: int = 20,
        price_days: int = 45,
    ) -> SignalBundle:
        company = company.strip()
        ticker = ticker.strip().upper()
        benchmark = benchmark.strip().upper()
        if not company or not ticker or not benchmark:
            raise ValueError("company, ticker and benchmark cannot be empty")
        cutoff_date = _normalise_cutoff(cutoff_date)
        if min(news_days, news_limit, price_days) < 1:
            raise ValueError("lookbacks and news limit must be at least 1")

        news = self.get_news_signals(
            company,
            ticker,
            cutoff_date,
            news_days=news_days,
            news_limit=news_limit,
        )
        market = self.get_market_signals(
            ticker,
            cutoff_date,
            benchmark=benchmark,
            price_days=price_days,
        )
        filing = self.get_filing_metadata(ticker, cutoff_date)

        return SignalBundle(
            company=company,
            ticker=ticker,
            cutoff_date=cutoff_date,
            benchmark=benchmark,
            signals=news.signals + market.signals + filing.signals,
            news_stats=news.news_stats or NewsStats(0, 0, 0, news_limit),
            evidence=news.evidence + filing.evidence,
            failures=news.failures + market.failures + filing.failures,
        )

    def get_news_signals(
        self,
        company: str,
        ticker: str,
        cutoff_date: datetime,
        *,
        news_days: int = 7,
        news_limit: int = 20,
    ) -> SignalResult:
        company = company.strip()
        ticker = ticker.strip().upper()
        cutoff_date = _normalise_cutoff(cutoff_date)
        if not company or not ticker:
            raise ValueError("company and ticker cannot be empty")
        if min(news_days, news_limit) < 1:
            raise ValueError("news lookback and limit must be at least 1")

        try:
            news_analysis = self._news_analyser.analyse(
                company,
                ticker=ticker,
                limit=news_limit,
                lookback_days=news_days,
                cutoff_date=cutoff_date,
            )
            return SignalResult(
                signals=tuple(calculate_news_signals(news_analysis, cutoff_date)),
                evidence=tuple(
                    EvidenceReference(
                        kind="news",
                        title=item.article.title,
                        url=item.article.url,
                        available_at=item.article.published_at,
                        excerpt=item.evidence,
                        sentiment=item.sentiment.label,
                        confidence=item.sentiment.confidence,
                    )
                    for item in news_analysis.articles
                    if item.article.published_at is not None
                ),
                failures=tuple(
                    SignalFailure("news", failure.stage, failure.reason, failure.url)
                    for failure in news_analysis.failures
                ),
                news_stats=NewsStats(
                    eligible=news_analysis.articles_eligible,
                    attempted=news_analysis.articles_attempted,
                    analysed=len(news_analysis.articles),
                    limit=news_analysis.analysis_limit,
                ),
            )
        except NewsProviderError as exc:
            return SignalResult(
                failures=(SignalFailure("news", "fetch", str(exc)),),
                news_stats=NewsStats(0, 0, 0, news_limit),
            )

    def get_market_signals(
        self,
        ticker: str,
        cutoff_date: datetime,
        *,
        benchmark: str = "SPY",
        price_days: int = 45,
    ) -> SignalResult:
        ticker = ticker.strip().upper()
        benchmark = benchmark.strip().upper()
        cutoff_date = _normalise_cutoff(cutoff_date)
        if not ticker or not benchmark:
            raise ValueError("ticker and benchmark cannot be empty")
        if price_days < 1:
            raise ValueError("price lookback must be at least 1")

        price_start = cutoff_date.date() - timedelta(days=price_days)
        price_end = cutoff_date.date()
        try:
            company_prices = self._price_provider.fetch(ticker, price_start, price_end)
            benchmark_prices = self._price_provider.fetch(
                benchmark, price_start, price_end
            )
            return SignalResult(
                signals=tuple(
                    calculate_market_signals(
                        company_prices,
                        benchmark_prices,
                        cutoff_date,
                        self._price_provider.source,
                    )
                )
            )
        except SignalProviderError as exc:
            return SignalResult(
                failures=(SignalFailure("prices", "fetch", str(exc)),)
            )

    def get_filing_metadata(
        self,
        ticker: str,
        cutoff_date: datetime,
    ) -> SignalResult:
        ticker = ticker.strip().upper()
        cutoff_date = _normalise_cutoff(cutoff_date)
        if not ticker:
            raise ValueError("ticker cannot be empty")

        try:
            filing = self._filing_provider.latest(
                ticker, cutoff_date, DEFAULT_FILING_FORMS
            )
            if filing is None:
                return SignalResult(
                    failures=(
                        SignalFailure(
                            "sec",
                            "availability",
                            "no supported filing was available by cutoff_date",
                        ),
                    )
                )
            return SignalResult(
                signals=tuple(calculate_filing_signals(filing, cutoff_date)),
                evidence=(
                    EvidenceReference(
                        kind="filing",
                        title=f"{filing.form} {filing.accession_number}",
                        url=filing.url,
                        available_at=filing.accepted_at,
                    ),
                ),
            )
        except SignalProviderError as exc:
            return SignalResult(failures=(SignalFailure("sec", "fetch", str(exc)),))


def _normalise_cutoff(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("cutoff_date must include a timezone")
    return value.astimezone(timezone.utc)
