from __future__ import annotations

from typing import Any

from news_signal.models import SentimentLabel, SentimentResult

LABELS: dict[str, SentimentLabel] = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
}


class HuggingFaceSentimentClassifier:
    def __init__(self, model_name: str, classifier: Any | None = None) -> None:
        self._model_name = model_name
        self._classifier = classifier

    def classify(self, title: str, content: str) -> SentimentResult:
        classifier = self._classifier or self._load_classifier()
        output = classifier(f"{title} {content}".strip(), truncation=True)[0]
        raw_label = str(output["label"]).lower()
        label = LABELS.get(raw_label)
        if label is None:
            raise ValueError(f"unsupported sentiment label: {raw_label}")
        return SentimentResult(label=label, confidence=float(output["score"]))

    def _load_classifier(self) -> Any:
        from transformers import pipeline

        self._classifier = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
        )
        return self._classifier
