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
    Signal,
    SignalBundle,
    SignalFailure,
    SignalProviderError,
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
        if cutoff_date.tzinfo is None:
            raise ValueError("cutoff_date must include a timezone")
        cutoff_date = cutoff_date.astimezone(timezone.utc)
        if min(news_days, news_limit, price_days) < 1:
            raise ValueError("lookbacks and news limit must be at least 1")

        signals: list[Signal] = []
        evidence: list[EvidenceReference] = []
        failures: list[SignalFailure] = []
        news_stats = NewsStats(0, 0, 0, news_limit)
        try:
            # get news and sentiment results
            news_analysis = self._news_analyser.analyse(
                company,
                ticker=ticker,
                limit=news_limit,
                lookback_days=news_days,
                cutoff_date=cutoff_date,
            )
            # calculate and store news signals - average sentiment of articles, article age, disagreement
            signals.extend(calculate_news_signals(news_analysis, cutoff_date))
            news_stats = NewsStats(
                eligible=news_analysis.articles_eligible,
                attempted=news_analysis.articles_attempted,
                analysed=len(news_analysis.articles),
                limit=news_analysis.analysis_limit,
            )
            evidence.extend(
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
            )
            failures.extend(
                SignalFailure("news", failure.stage, failure.reason, failure.url)
                for failure in news_analysis.failures
            )
        except NewsProviderError as exc:
            failures.append(SignalFailure("news", "fetch", str(exc)))

        price_start = cutoff_date.date() - timedelta(days=price_days)
        price_end = cutoff_date.date()
        try:
            # fetch price data for company
            company_prices = self._price_provider.fetch(ticker, price_start, price_end)
            # default to SPY - how US market performs as a comparison
            benchmark_prices = self._price_provider.fetch(benchmark, price_start, price_end)
            # market signals - momentum, volatility, volume, benchmark relative return
            signals.extend(
                calculate_market_signals(
                    company_prices,
                    benchmark_prices,
                    cutoff_date,
                    self._price_provider.source,
                )
            )
        except SignalProviderError as exc:
            failures.append(SignalFailure("prices", "fetch", str(exc)))

        # sec filings - good for understanding market reactions, recent developments and long term analysis
        # does not actually extract filings
        try:
            filing = self._filing_provider.latest(
                ticker, cutoff_date, DEFAULT_FILING_FORMS
            )
            if filing is None:
                failures.append(
                    SignalFailure(
                        "sec",
                        "availability",
                        "no supported filing was available by cutoff_date",
                    )
                )
            else:
                signals.extend(calculate_filing_signals(filing, cutoff_date))
                evidence.append(
                    EvidenceReference(
                        kind="filing",
                        title=f"{filing.form} {filing.accession_number}",
                        url=filing.url,
                        available_at=filing.accepted_at,
                    )
                )
        except SignalProviderError as exc:
            failures.append(SignalFailure("sec", "fetch", str(exc)))

        return SignalBundle(
            company=company,
            ticker=ticker,
            cutoff_date=cutoff_date,
            benchmark=benchmark,
            signals=tuple(signals),
            news_stats=news_stats,
            evidence=tuple(evidence),
            failures=tuple(failures),
        )
