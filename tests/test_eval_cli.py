import json

import pytest

from eval import cli


class FakeClassifier:
    model_revision = "revision"

    def load(self) -> None:
        pass


class FakeReport:
    def to_dict(self) -> dict[str, object]:
        return {"accuracy": 0.75}


def test_evaluation_cli_does_not_require_news_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    classifier = FakeClassifier()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_default_model", lambda system: "test/model")
    monkeypatch.setattr(
        cli, "_build_system", lambda system, model: (object(), classifier, "ignored")
    )
    monkeypatch.setattr(cli, "load_finentity", lambda path: object())
    monkeypatch.setattr(cli, "evaluate_finentity", lambda *args, **kwargs: FakeReport())

    exit_code = cli.main(["--system", "v1", "--dataset", "fixture.json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["accuracy"] == 0.75
