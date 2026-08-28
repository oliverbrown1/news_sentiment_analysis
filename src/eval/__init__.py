from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal, Protocol, cast

SentimentLabel = Literal["positive", "neutral", "negative"]
LABELS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")
FINENTITY_SOURCE = "https://github.com/yixuantt/FinEntity"
FINENTITY_REVISION = "3b6cedc5485b669c2ed168f1d949f517636eb7b8"


class EvaluationDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    label: SentimentLabel
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class SentimentSystem(Protocol):
    def predict(self, target: str, text: str) -> SentimentPrediction: ...


@dataclass(frozen=True, slots=True)
class EntityAnnotation:
    annotation_id: int
    target: str
    label: SentimentLabel
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class FinEntityParagraph:
    paragraph_id: int
    content: str
    annotations: tuple[EntityAnnotation, ...]


@dataclass(frozen=True, slots=True)
class FinEntityDataset:
    path: str
    sha256: str
    paragraphs: tuple[FinEntityParagraph, ...]
    span_mismatches: int
    duplicate_annotations: int
    conflicting_target_labels: int


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    paragraph_id: int
    annotation_id: int
    target: str
    reason: str


@dataclass(frozen=True, slots=True)
class SentimentEvaluationReport:
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
    failures: tuple[EvaluationFailure, ...]
    environment: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ScoredAnnotation:
    paragraph: FinEntityParagraph
    annotation: EntityAnnotation
    prediction: SentimentPrediction


def load_finentity(path: Path) -> FinEntityDataset:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvaluationDataError(f"could not read FinEntity dataset: {path}") from exc

    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvaluationDataError(f"invalid FinEntity JSON: {path}") from exc
    if not isinstance(data, list):
        raise EvaluationDataError("FinEntity root must be a JSON array")

    paragraphs: list[FinEntityParagraph] = []
    span_mismatches = 0
    duplicate_annotations = 0
    conflicting_target_labels = 0

    for paragraph_id, item in enumerate(data):
        if not isinstance(item, dict):
            raise EvaluationDataError(f"paragraph {paragraph_id} must be an object")
        content = item.get("content")
        raw_annotations = item.get("annotations")
        if not isinstance(content, str) or not content.strip():
            raise EvaluationDataError(f"paragraph {paragraph_id} has invalid content")
        if not isinstance(raw_annotations, list) or not raw_annotations:
            raise EvaluationDataError(f"paragraph {paragraph_id} has no annotations")

        annotations: list[EntityAnnotation] = []
        seen: set[tuple[str, SentimentLabel]] = set()
        labels_by_target: dict[str, set[SentimentLabel]] = {}

        for annotation_id, item_annotation in enumerate(raw_annotations):
            if not isinstance(item_annotation, dict):
                raise EvaluationDataError(
                    f"annotation {paragraph_id}:{annotation_id} must be an object"
                )
            target = item_annotation.get("value")
            raw_label = item_annotation.get("label")
            start = item_annotation.get("start")
            end = item_annotation.get("end")
            if not isinstance(target, str) or not target.strip():
                raise EvaluationDataError(
                    f"annotation {paragraph_id}:{annotation_id} has invalid target"
                )
            if not isinstance(raw_label, str) or raw_label.lower() not in LABELS:
                raise EvaluationDataError(
                    f"annotation {paragraph_id}:{annotation_id} has invalid label"
                )
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(content)
            ):
                raise EvaluationDataError(
                    f"annotation {paragraph_id}:{annotation_id} has invalid span"
                )

            label = cast(SentimentLabel, raw_label.lower())
            annotation = EntityAnnotation(annotation_id, target.strip(), label, start, end)
            annotations.append(annotation)

            key = (annotation.target.casefold(), label)
            if key in seen:
                duplicate_annotations += 1
            seen.add(key)
            labels_by_target.setdefault(annotation.target.casefold(), set()).add(label)
            if content[start:end] != target:
                span_mismatches += 1

        conflicting_target_labels += sum(
            len(target_labels) > 1 for target_labels in labels_by_target.values()
        )
        paragraphs.append(FinEntityParagraph(paragraph_id, content, tuple(annotations)))

    return FinEntityDataset(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        paragraphs=tuple(paragraphs),
        span_mismatches=span_mismatches,
        duplicate_annotations=duplicate_annotations,
        conflicting_target_labels=conflicting_target_labels,
    )


def evaluate_finentity(
    system: SentimentSystem,
    dataset: FinEntityDataset,
    *,
    system_name: str,
    model_name: str,
    model_revision: str | None = None,
    model_load_seconds: float | None = None,
    target_usage: str,
) -> SentimentEvaluationReport:
    scored: list[_ScoredAnnotation] = []
    failures: list[EvaluationFailure] = []
    inference_times: list[float] = []

    for paragraph in dataset.paragraphs:
        for annotation in paragraph.annotations:
            started = perf_counter()
            try:
                prediction = system.predict(annotation.target, paragraph.content)
            except (RuntimeError, ValueError) as exc:
                failures.append(
                    EvaluationFailure(
                        paragraph.paragraph_id,
                        annotation.annotation_id,
                        annotation.target,
                        str(exc),
                    )
                )
                continue
            inference_times.append(perf_counter() - started)
            scored.append(_ScoredAnnotation(paragraph, annotation, prediction))

    if not scored:
        raise RuntimeError("sentiment evaluation produced no predictions")

    all_metrics = _classification_metrics(scored)
    label_distribution = Counter(
        annotation.label
        for paragraph in dataset.paragraphs
        for annotation in paragraph.annotations
    )
    predictions = Counter(item.prediction.label for item in scored)

    return SentimentEvaluationReport(
        schema_version=2,
        generated_at=datetime.now(timezone.utc).isoformat(),
        task={
            "name": "entity-level financial sentiment",
            "unit": "entity annotation",
            "input": "target entity and paragraph content",
            "target_usage": target_usage,
        },
        dataset={
            "name": "FinEntity",
            "path": dataset.path,
            "source": FINENTITY_SOURCE,
            "revision": FINENTITY_REVISION,
            "sha256": dataset.sha256,
            "paragraphs": len(dataset.paragraphs),
            "annotations": sum(label_distribution.values()),
            "label_distribution": dict(label_distribution),
            "diagnostics": {
                "span_mismatches": dataset.span_mismatches,
                "duplicate_annotations": dataset.duplicate_annotations,
                "targets_with_conflicting_labels": dataset.conflicting_target_labels,
            },
        },
        model={"system": system_name, "name": model_name, "revision": model_revision},
        metrics={
            **all_metrics,
            "prediction_distribution": dict(predictions),
        },
        slices=_slice_metrics(scored),
        calibration=_calibration_metrics(scored),
        error_analysis=_error_analysis(scored),
        latency=_latency_metrics(inference_times, model_load_seconds),
        failures=tuple(failures),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
    )


def _classification_metrics(scored: list[_ScoredAnnotation]) -> dict[str, object]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_indexes = {label: index for index, label in enumerate(LABELS)}
    for item in scored:
        matrix[label_indexes[item.annotation.label]][label_indexes[item.prediction.label]] += 1

    total = len(scored)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []

    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = _divide(true_positive, predicted)
        recall = _divide(true_positive, support)
        f1 = _divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
        f1_scores.append(f1)

    accuracy = correct / total

    return {
        "evaluated_annotations": total,
        "accuracy": accuracy,
        "macro_f1": mean(f1_scores),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "rows": matrix},
    }


def _slice_metrics(scored: list[_ScoredAnnotation]) -> dict[str, object]:
    items = [
        item
        for item in scored
        if len({annotation.label for annotation in item.paragraph.annotations}) > 1
    ]
    if not items:
        return {"mixed_sentiment_paragraphs": {"annotations": 0}}

    metrics = _classification_metrics(items)
    return {
        "mixed_sentiment_paragraphs": {
            "annotations": len(items),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
    }


def _calibration_metrics(
    scored: list[_ScoredAnnotation], bins: int = 10
) -> dict[str, object]:
    bucketed: list[list[_ScoredAnnotation]] = [[] for _ in range(bins)]
    for item in scored:
        index = min(int(item.prediction.confidence * bins), bins - 1)
        bucketed[index].append(item)

    calibration_bins: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for index, items in enumerate(bucketed):
        if not items:
            continue
        accuracy = mean(
            item.annotation.label == item.prediction.label for item in items
        )
        average_confidence = mean(item.prediction.confidence for item in items)
        expected_calibration_error += (
            len(items) / len(scored) * abs(accuracy - average_confidence)
        )
        calibration_bins.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(items),
                "accuracy": accuracy,
                "average_confidence": average_confidence,
            }
        )

    correct_confidences = [
        item.prediction.confidence
        for item in scored
        if item.annotation.label == item.prediction.label
    ]
    incorrect_confidences = [
        item.prediction.confidence
        for item in scored
        if item.annotation.label != item.prediction.label
    ]
    return {
        "expected_calibration_error": expected_calibration_error,
        "mean_confidence": mean(item.prediction.confidence for item in scored),
        "mean_confidence_when_correct": mean(correct_confidences)
        if correct_confidences
        else None,
        "mean_confidence_when_incorrect": mean(incorrect_confidences)
        if incorrect_confidences
        else None,
        "bins": calibration_bins,
    }


def _error_analysis(scored: list[_ScoredAnnotation]) -> dict[str, object]:
    errors = [
        item
        for item in scored
        if item.annotation.label != item.prediction.label
    ]
    confusion_pairs = Counter(
        f"{item.annotation.label}->{item.prediction.label}" for item in errors
    )
    highest_confidence = sorted(
        errors, key=lambda item: item.prediction.confidence, reverse=True
    )[:20]
    return {
        "misclassified_annotations": len(errors),
        "confusion_pairs": dict(confusion_pairs.most_common()),
        "highest_confidence_errors": [
            {
                "paragraph_id": item.paragraph.paragraph_id,
                "annotation_id": item.annotation.annotation_id,
                "target": item.annotation.target,
                "expected": item.annotation.label,
                "predicted": item.prediction.label,
                "confidence": item.prediction.confidence,
                "content": item.paragraph.content,
            }
            for item in highest_confidence
        ],
    }


def _latency_metrics(
    inference_times: list[float], model_load_seconds: float | None
) -> dict[str, object]:
    milliseconds = [duration * 1000 for duration in inference_times]
    return {
        "model_load_seconds": model_load_seconds,
        "annotations_succeeded": len(milliseconds),
        "mean_ms_per_annotation": mean(milliseconds),
    }


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
