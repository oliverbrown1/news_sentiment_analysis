import json

import pytest

from news_signal_v2.entrypoints import cli
from news_signal_v2.models import AnalysisResult


class FakePipeline:
    def __init__(self) -> None:
        self.arguments = None

    def analyse(self, **kwargs) -> AnalysisResult:
        self.arguments = kwargs
        return AnalysisResult(company=kwargs["company"], ticker=kwargs["ticker"], articles=())


def test_v2_cli_runs_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = FakePipeline()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_pipeline", lambda settings: pipeline)

    exit_code = cli.main(
        ["analyse", "--company", "Example Ltd", "--ticker", "EXM", "--days", "14"]
    )

    assert exit_code == 0
    assert pipeline.arguments["ticker"] == "EXM"
    assert json.loads(capsys.readouterr().out)["company"] == "Example Ltd"
