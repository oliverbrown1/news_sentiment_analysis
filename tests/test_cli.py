import json

import pytest

from news_signal import cli
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
