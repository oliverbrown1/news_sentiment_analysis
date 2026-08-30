from __future__ import annotations

import ast
import csv
import hashlib
import io
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from eval.metrics import (
    calibration_metrics,
    classification_metrics,
    environment_versions,
    latency_metrics,
)
from eval.models import (
    ClassificationLabel,
    ClassificationPrediction,
    EvaluationDataError,
    ScoredPrediction,
)

MarketDirection = ClassificationLabel
MarketPrediction = ClassificationPrediction
FINMARBA_SOURCE = (
    "https://huggingface.co/datasets/baptle/financial_headlines_market_based"
)
FINMARBA_REVISION = "d4ac449b57cb7dddfd5d8e05fd1326f8cb03fa29"
FINMARBA_SHA256 = "554afe421d62f6a93f342ba10def593a485120c86d5e5074477d03dd95258a8c"
REQUIRED_COLUMNS = {
    "Date",
    "Title",
    "Tickers",
    "Sentiment",
    "Global Sentiment",
    "Pct_Change",
}
LABEL_BY_VALUE: dict[int, MarketDirection] = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}


class MarketSystem(Protocol):
    def predict(
        self, as_of: date, ticker: str, headline: str
    ) -> MarketPrediction: ...


@dataclass(frozen=True, slots=True)
class FinMarBaExample:
    example_id: int
    source_row: int
    as_of: date
    ticker: str
    headline: str
    label: MarketDirection
    return_pct: float | None


@dataclass(frozen=True, slots=True)
class FinMarBaDataset:
    path: str
    sha256: str
    source_rows: int
    examples: tuple[FinMarBaExample, ...]
    missing_ticker_labels: int
    missing_ticker_returns: int
    duplicate_examples: int


@dataclass(frozen=True, slots=True)
class MarketEvaluationFailure:
    example_id: int
    ticker: str
    reason: str


@dataclass(frozen=True, slots=True)
class MarketEvaluationReport:
    schema_version: int
    generated_at: str
    task: dict[str, str]
    dataset: dict[str, object]
    model: dict[str, object]
    metrics: dict[str, object]
    slices: dict[str, object]
    calibration: dict[str, object]
    error_analysis: dict[str, object]
    latency: dict[str, object]
    failures: tuple[MarketEvaluationFailure, ...]
    environment: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ScoredExample:
    example: FinMarBaExample
    prediction: MarketPrediction


def load_finmarba(path: Path) -> FinMarBaDataset:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvaluationDataError(f"could not read FinMarBa dataset: {path}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationDataError(f"invalid FinMarBa CSV encoding: {path}") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = set(reader.fieldnames or ())
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise EvaluationDataError(f"FinMarBa is missing columns: {names}")

    examples: list[FinMarBaExample] = []
    missing_ticker_labels = 0
    missing_ticker_returns = 0
    duplicate_examples = 0
    seen: set[tuple[date, str, str]] = set()
    source_rows = 0

    for source_row, row in enumerate(reader, start=2):
        source_rows += 1
        try:
            as_of = date.fromisoformat(row["Date"].strip())
            headline = row["Title"].strip()
            tickers = _parse_tickers(row["Tickers"])
            sentiments = _parse_mapping(row["Sentiment"], "Sentiment")
            returns = _parse_mapping(row["Pct_Change"], "Pct_Change")
            _label(row["Global Sentiment"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationDataError(
                f"invalid FinMarBa row {source_row}: {exc}"
            ) from exc

        if not headline:
            raise EvaluationDataError(f"invalid FinMarBa row {source_row}: empty title")

        for ticker in tickers:
            raw_label = sentiments.get(ticker)
            if raw_label is None:
                missing_ticker_labels += 1
                continue

            label = _label(raw_label)
            raw_return = returns.get(ticker)
            return_pct = _return_value(raw_return) if raw_return is not None else None
            if return_pct is None:
                missing_ticker_returns += 1

            key = (as_of, ticker, headline.casefold())
            if key in seen:
                duplicate_examples += 1
            seen.add(key)
            examples.append(
                FinMarBaExample(
                    example_id=len(examples),
                    source_row=source_row,
                    as_of=as_of,
                    ticker=ticker,
                    headline=headline,
                    label=label,
                    return_pct=return_pct,
                )
            )

    if not examples:
        raise EvaluationDataError("FinMarBa contains no labelled ticker examples")

    return FinMarBaDataset(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        source_rows=source_rows,
        examples=tuple(examples),
        missing_ticker_labels=missing_ticker_labels,
        missing_ticker_returns=missing_ticker_returns,
        duplicate_examples=duplicate_examples,
    )


def evaluate_finmarba(
    system: MarketSystem,
    dataset: FinMarBaDataset,
    *,
    system_name: str,
    model_name: str,
    model_revision: str | None = None,
    model_load_seconds: float | None = None,
) -> MarketEvaluationReport:
    scored: list[_ScoredExample] = []
    failures: list[MarketEvaluationFailure] = []
    inference_times: list[float] = []

    for example in dataset.examples:
        started = perf_counter()
        try:
            prediction = system.predict(
                example.as_of, example.ticker, example.headline
            )
        except (RuntimeError, ValueError) as exc:
            failures.append(
                MarketEvaluationFailure(example.example_id, example.ticker, str(exc))
            )
            continue
        inference_times.append(perf_counter() - started)
        scored.append(_ScoredExample(example, prediction))

    if not scored:
        raise RuntimeError("market evaluation produced no predictions")

    common_scored = [
        ScoredPrediction(item.example.label, item.prediction) for item in scored
    ]
    label_distribution = Counter(example.label for example in dataset.examples)
    prediction_distribution = Counter(item.prediction.label for item in scored)

    return MarketEvaluationReport(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        task={
            "name": "ticker-level market direction",
            "unit": "dated headline and ticker",
            "input": "date, ticker, and financial headline",
            "target": "FinMarBa market-derived direction",
        },
        dataset={
            "name": "FinMarBa released subset",
            "path": dataset.path,
            "source": FINMARBA_SOURCE,
            "revision": FINMARBA_REVISION,
            "sha256": dataset.sha256,
            "source_rows": dataset.source_rows,
            "examples": len(dataset.examples),
            "date_start": min(item.as_of for item in dataset.examples).isoformat(),
            "date_end": max(item.as_of for item in dataset.examples).isoformat(),
            "label_distribution": dict(label_distribution),
            "diagnostics": {
                "missing_ticker_labels": dataset.missing_ticker_labels,
                "missing_ticker_returns": dataset.missing_ticker_returns,
                "duplicate_examples": dataset.duplicate_examples,
            },
        },
        model={"system": system_name, "name": model_name, "revision": model_revision},
        metrics={
            **classification_metrics(common_scored, count_key="evaluated_examples"),
            "prediction_distribution": dict(prediction_distribution),
        },
        slices={},
        calibration=calibration_metrics(common_scored),
        error_analysis=_error_analysis(scored),
        latency=latency_metrics(
            inference_times,
            model_load_seconds,
            count_key="examples_succeeded",
            mean_key="mean_ms_per_example",
        ),
        failures=tuple(failures),
        environment=environment_versions(),
    )


def _parse_tickers(raw: str) -> list[str]:
    parsed = _literal(raw, "Tickers")
    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], str)
        and parsed[0].strip().startswith("[")
    ):
        parsed = _literal(parsed[0], "Tickers")
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Tickers must be a non-empty list")
    if not all(isinstance(ticker, str) and ticker.strip() for ticker in parsed):
        raise ValueError("Tickers contains an invalid ticker")
    return [ticker.strip() for ticker in parsed]


def _parse_mapping(raw: str, field: str) -> dict[str, object]:
    parsed = _literal(raw, field)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in parsed):
        raise ValueError(f"{field} contains an invalid ticker")
    return cast(dict[str, object], parsed)


def _literal(raw: str, field: str) -> object:
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{field} contains invalid structured data") from exc


def _label(raw: object) -> MarketDirection:
    if isinstance(raw, str):
        try:
            raw = int(raw.strip())
        except ValueError as exc:
            raise ValueError("sentiment label must be -1, 0, or 1") from exc
    if not isinstance(raw, int) or isinstance(raw, bool) or raw not in LABEL_BY_VALUE:
        raise ValueError("sentiment label must be -1, 0, or 1")
    return LABEL_BY_VALUE[raw]


def _return_value(raw: object) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError("Pct_Change contains an invalid return")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Pct_Change contains a non-finite return")
    return value


def _error_analysis(scored: list[_ScoredExample]) -> dict[str, object]:
    errors = [item for item in scored if item.example.label != item.prediction.label]
    confusion_pairs = Counter(
        f"{item.example.label}->{item.prediction.label}" for item in errors
    )
    highest_confidence = sorted(
        errors, key=lambda item: item.prediction.confidence, reverse=True
    )[:20]
    return {
        "misclassified_examples": len(errors),
        "confusion_pairs": dict(confusion_pairs.most_common()),
        "highest_confidence_errors": [
            {
                "example_id": item.example.example_id,
                "date": item.example.as_of.isoformat(),
                "ticker": item.example.ticker,
                "headline": item.example.headline,
                "expected": item.example.label,
                "predicted": item.prediction.label,
                "confidence": item.prediction.confidence,
                "return_pct": item.example.return_pct,
            }
            for item in highest_confidence
        ],
    }
