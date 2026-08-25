import json
from pathlib import Path

import pytest

from news_signal.evaluation import (
    EvaluationDataError,
    evaluate_finentity,
    load_finentity,
)
from news_signal.models import SentimentResult


class FakeClassifier:
    def classify(self, title: str, content: str) -> SentimentResult:
        if content == "Alpha rises.":
            return SentimentResult("positive", 0.9)
        return SentimentResult("neutral", 0.6)


def write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "content": "Alpha rises.",
                    "annotations": [
                        {
                            "value": "Alpha",
                            "label": "Positive",
                            "start": 0,
                            "end": 5,
                        }
                    ],
                },
                {
                    "content": "Beta falls while Gamma is unchanged.",
                    "annotations": [
                        {
                            "value": "Beta",
                            "label": "Negative",
                            "start": 0,
                            "end": 4,
                        },
                        {
                            "value": "Gamma",
                            "label": "Neutral",
                            "start": 17,
                            "end": 22,
                        },
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_finentity_evaluation_reports_class_and_slice_metrics(tmp_path: Path) -> None:
    path = tmp_path / "finentity.json"
    write_dataset(path)

    report = evaluate_finentity(
        FakeClassifier(),
        load_finentity(path),
        model_name="test/model",
        model_revision="abc123",
        model_load_seconds=0.25,
    ).to_dict()

    metrics = report["metrics"]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(5 / 9)
    assert metrics["per_class"]["negative"]["recall"] == 0
    assert metrics["confusion_matrix"]["rows"] == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert report["slices"]["mixed_sentiment_paragraphs"]["accuracy"] == 0.5
    assert report["calibration"]["expected_calibration_error"] == pytest.approx(0.1)
    assert report["error_analysis"]["misclassified_annotations"] == 1
    assert report["error_analysis"]["highest_confidence_errors"][0]["target"] == "Beta"
    assert report["model"] == {"name": "test/model", "revision": "abc123"}
    assert report["schema_version"] == 2


def test_finentity_loader_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "finentity.json"
    path.write_text(
        json.dumps(
            [
                {
                    "content": "Alpha rises.",
                    "annotations": [
                        {"value": "Alpha", "label": "Bullish", "start": 0, "end": 5}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="invalid label"):
        load_finentity(path)


def test_vendored_finentity_dataset_is_pinned() -> None:
    dataset = load_finentity(Path("data/finentity.json"))

    assert len(dataset.paragraphs) == 979
    assert sum(len(item.annotations) for item in dataset.paragraphs) == 2131
    assert dataset.sha256 == (
        "3208667de69383120b0380aebeaabe360669eda72269c33e1c8b09d63df55463"
    )
