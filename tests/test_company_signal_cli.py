import json
from datetime import datetime

import pytest

from company_signals.entrypoints import cli


class FakeResult:
    def __init__(self) -> None:
        self.verbose: bool | None = None

    def to_dict(self, *, verbose: bool = False) -> dict[str, object]:
        self.verbose = verbose
        return {"verbose": verbose}


class FakePipeline:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}
        self.result = FakeResult()

    def collect(self, **kwargs: object) -> FakeResult:
        self.arguments = kwargs
        return self.result


def test_company_signal_cli_passes_options_and_controls_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "build_pipeline", lambda: pipeline)

    exit_code = cli.main(
        [
            "collect",
            "--company",
            "Example Ltd",
            "--ticker",
            "EXM",
            "--cutoff-date",
            "2024-02-20",
            "--news-days",
            "14",
            "--news-limit",
            "10",
            "--price-days",
            "60",
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert pipeline.arguments["company"] == "Example Ltd"
    assert pipeline.arguments["news_days"] == 14
    assert pipeline.arguments["news_limit"] == 10
    assert pipeline.arguments["price_days"] == 60
    assert isinstance(pipeline.arguments["cutoff_date"], datetime)
    assert pipeline.result.verbose is True
    assert json.loads(capsys.readouterr().out) == {"verbose": True}
