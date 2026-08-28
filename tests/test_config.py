import pytest

from news_signal_v1.config import ConfigurationError, Settings


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", "14")
    monkeypatch.setenv("SENTIMENT_MODEL", "test/model")

    settings = Settings.from_env()

    assert settings.news_api_key == "test-key"
    assert settings.lookback_days == 14
    assert settings.sentiment_model == "test/model"


def test_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="NEWS_API_KEY is required"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["zero", "0", "-2"])
def test_settings_rejects_invalid_lookback(
    monkeypatch: pytest.MonkeyPatch, tmp_path, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", value)

    with pytest.raises(ConfigurationError, match="NEWS_LOOKBACK_DAYS"):
        Settings.from_env()
