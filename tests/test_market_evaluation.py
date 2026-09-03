import csv
from datetime import date
from pathlib import Path

import pytest

from eval.market_eval import (
    FINMARBA_SHA256,
    MarketPrediction,
    evaluate_finmarba,
    load_finmarba,
)
from eval.models import EvaluationDataError


class FakeMarketSystem:
    def predict(
        self, cutoff_date: date, ticker: str, headline: str
    ) -> MarketPrediction:
        del cutoff_date, headline
        if ticker == "AAA":
            return MarketPrediction("positive", 0.9)
        return MarketPrediction("neutral", 0.6)


def write_dataset(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "Date",
                "Title",
                "Tickers",
                "Sentiment",
                "Global Sentiment",
                "Pct_Change",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Date": "2024-01-02",
                "Title": "Alpha rises",
                "Tickers": "[\"['AAA']\"]",
                "Sentiment": "{'AAA': 1}",
                "Global Sentiment": "1",
                "Pct_Change": "{'AAA': 0.02}",
            }
        )
        writer.writerow(
            {
                "Date": "2024-01-03",
                "Title": "Beta falls while Gamma is unchanged",
                "Tickers": "[\"['BBB', 'CCC']\"]",
                "Sentiment": "{'BBB': -1, 'CCC': 0}",
                "Global Sentiment": "0",
                "Pct_Change": "{'BBB': -0.03, 'CCC': 0.001}",
            }
        )


def test_market_evaluation_reuses_classification_metrics(tmp_path: Path) -> None:
    path = tmp_path / "finmarba.csv"
    write_dataset(path)

    report = evaluate_finmarba(
        FakeMarketSystem(),
        load_finmarba(path),
        system_name="baseline",
        model_name="test/model",
        model_revision="abc123",
        model_load_seconds=0.25,
    ).to_dict()

    assert report["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert report["metrics"]["macro_f1"] == pytest.approx(5 / 9)
    assert report["metrics"]["per_class"]["negative"]["recall"] == 0
    assert report["metrics"]["confusion_matrix"]["rows"] == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert report["calibration"]["expected_calibration_error"] == pytest.approx(0.1)
    assert report["error_analysis"]["misclassified_examples"] == 1
    assert report["error_analysis"]["highest_confidence_errors"][0]["ticker"] == "BBB"
    assert report["latency"]["examples_succeeded"] == 3


def test_finmarba_loader_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("Date,Title\n2024-01-02,Headline\n", encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="missing columns"):
        load_finmarba(path)


def test_vendored_finmarba_dataset_is_pinned() -> None:
    dataset = load_finmarba(Path("data/finmarba.csv"))

    assert dataset.source_rows == 8142
    assert len(dataset.examples) == 9978
    assert dataset.missing_ticker_labels == 558
    assert dataset.missing_ticker_returns == 0
    assert dataset.sha256 == FINMARBA_SHA256
    assert min(item.cutoff_date for item in dataset.examples) == date(2010, 1, 4)
    assert max(item.cutoff_date for item in dataset.examples) == date(2011, 12, 30)
