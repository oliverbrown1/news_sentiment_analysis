from __future__ import annotations

import os

from company_signals.pipeline import CompanySignalPipeline
from company_signals.providers import SecFilingProvider, YFinancePriceProvider
from news_signal_v2.application import build_pipeline as build_news_pipeline
from news_signal_v2.config import ConfigurationError, Settings


def build_pipeline(
    settings: Settings | None = None,
    sec_user_agent: str | None = None,
) -> CompanySignalPipeline:
    settings = settings or Settings.from_env()
    sec_user_agent = (sec_user_agent or os.getenv("SEC_USER_AGENT", "")).strip()
    if not sec_user_agent:
        raise ConfigurationError(
            "SEC_USER_AGENT is required and should include a contact email"
        )
    return CompanySignalPipeline(
        news_analyser=build_news_pipeline(settings),
        price_provider=YFinancePriceProvider(),
        filing_provider=SecFilingProvider(sec_user_agent),
    )
