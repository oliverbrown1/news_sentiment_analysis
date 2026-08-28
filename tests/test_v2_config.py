import pytest

from news_signal_v2.config import ConfigurationError, Settings


def test_v2_settings_reads_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", "14")
    monkeypatch.setenv("V2_SENTIMENT_MODEL", "test/model")
    monkeypatch.setenv("NEWS_DOMAINS", "reuters.com, ft.com")

    settings = Settings.from_env()

    assert settings.news_api_key == "test-key"
    assert settings.lookback_days == 14
    assert settings.sentiment_model == "test/model"
    assert settings.news_domains == ("reuters.com", "ft.com")


def test_v2_settings_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="NEWS_API_KEY is required"):
        Settings.from_env()
