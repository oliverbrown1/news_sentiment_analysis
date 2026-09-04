import pytest

from market_signal_agent.config import DEFAULT_MODEL, get_model


def test_agent_model_reads_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MARKET_SIGNAL_MODEL", "test-model")

    assert get_model() == "test-model"


def test_agent_model_uses_default_when_environment_value_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MARKET_SIGNAL_MODEL", "")

    assert get_model() == DEFAULT_MODEL
