from __future__ import annotations

import platform
from importlib.metadata import version
from statistics import mean

from eval.models import LABELS, ScoredPrediction


def classification_metrics(
    scored: list[ScoredPrediction], *, count_key: str
) -> dict[str, object]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_indexes = {label: index for index, label in enumerate(LABELS)}
    for item in scored:
        matrix[label_indexes[item.expected]][label_indexes[item.prediction.label]] += 1

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

    return {
        count_key: total,
        "accuracy": correct / total,
        "macro_f1": mean(f1_scores),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "rows": matrix},
    }


def calibration_metrics(
    scored: list[ScoredPrediction], bins: int = 10
) -> dict[str, object]:
    bucketed: list[list[ScoredPrediction]] = [[] for _ in range(bins)]
    for item in scored:
        index = min(int(item.prediction.confidence * bins), bins - 1)
        bucketed[index].append(item)

    calibration_bins: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    for index, items in enumerate(bucketed):
        if not items:
            continue
        accuracy = mean(item.expected == item.prediction.label for item in items)
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
        if item.expected == item.prediction.label
    ]
    incorrect_confidences = [
        item.prediction.confidence
        for item in scored
        if item.expected != item.prediction.label
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


def latency_metrics(
    inference_times: list[float],
    model_load_seconds: float | None,
    *,
    count_key: str,
    mean_key: str,
) -> dict[str, object]:
    milliseconds = [duration * 1000 for duration in inference_times]
    return {
        "model_load_seconds": model_load_seconds,
        count_key: len(milliseconds),
        mean_key: mean(milliseconds),
    }


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
    }


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
