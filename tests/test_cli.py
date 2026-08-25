import json

import pytest

from news_signal.entrypoints import cli
from news_signal.models import AnalysisResult


class FakePipeline:
    def __init__(self) -> None:
        self.arguments = None

    def analyse(self, company: str, limit: int, lookback_days: int) -> AnalysisResult:
        self.arguments = (company, limit, lookback_days)
        return AnalysisResult(company=company, articles=())


def test_cli_runs_pipeline_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = FakePipeline()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_pipeline", lambda settings: pipeline)

    exit_code = cli.main(
        ["analyse", "--company", "Example Ltd", "--limit", "2", "--days", "14"]
    )

    assert exit_code == 0
    assert pipeline.arguments == ("Example Ltd", 2, 14)
    assert json.loads(capsys.readouterr().out)["company"] == "Example Ltd"


def test_evaluation_cli_does_not_require_news_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeClassifier:
        model_revision = "revision"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            pass

    class FakeReport:
        def to_dict(self) -> dict[str, object]:
            return {"accuracy": 0.75}

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_finentity", lambda path: object())
    monkeypatch.setattr(cli, "HuggingFaceSentimentClassifier", FakeClassifier)
    monkeypatch.setattr(cli, "evaluate_finentity", lambda *args, **kwargs: FakeReport())

    exit_code = cli.main(
        ["evaluate-sentiment", "--dataset", "fixture.json", "--model", "test/model"]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["accuracy"] == 0.75
