from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClassificationLabel = Literal["positive", "neutral", "negative"]
LABELS: tuple[ClassificationLabel, ...] = ("negative", "neutral", "positive")


class EvaluationDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    label: ClassificationLabel
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    expected: ClassificationLabel
    prediction: ClassificationPrediction
