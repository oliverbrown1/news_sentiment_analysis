from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EvidenceSource = Literal["headline", "news", "market", "filing"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str
    cutoff_date: datetime
    company: str | None = None
    headline: str | None = None
    headline_source: str | None = None
    headline_url: str | None = None
    forecast_horizon: str = "next trading day"
    benchmark: str = "SPY"

    @field_validator("ticker", "forecast_horizon", "benchmark")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("company", "headline", "headline_source", "headline_url")
    @classmethod
    def empty_optional_text(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("ticker", "benchmark")
    @classmethod
    def uppercase_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("cutoff_date")
    @classmethod
    def normalise_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("cutoff_date must include a timezone")
        return value.astimezone(timezone.utc)


class EvidenceCitation(BaseModel):
    source: EvidenceSource
    reference: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MarketForecast(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)

    ticker: str = Field(min_length=1)
    forecast_horizon: str = Field(min_length=1)
    predicted_return: float = Field(ge=-1)
    thesis: str = Field(min_length=1)
    risks: list[str]
    evidence: list[EvidenceCitation]

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, value: str) -> str:
        return value.upper()
