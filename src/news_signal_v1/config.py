from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


class ConfigurationError(ValueError):
    pass


def load_sentiment_model_name() -> str:
    load_dotenv(Path.cwd() / ".env")
    model = os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise ConfigurationError("SENTIMENT_MODEL cannot be empty")
    return model


@dataclass(frozen=True, slots=True)
class Settings:
    news_api_key: str
    lookback_days: int = 7
    sentiment_model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(Path.cwd() / ".env")

        api_key = os.getenv("NEWS_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("NEWS_API_KEY is required")

        raw_days = os.getenv("NEWS_LOOKBACK_DAYS", "7")
        try:
            lookback_days = int(raw_days)
        except ValueError as exc:
            raise ConfigurationError("NEWS_LOOKBACK_DAYS must be an integer") from exc
        if lookback_days < 1:
            raise ConfigurationError("NEWS_LOOKBACK_DAYS must be at least 1")

        return cls(api_key, lookback_days, load_sentiment_model_name())
