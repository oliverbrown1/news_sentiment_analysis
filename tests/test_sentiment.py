import pytest

from news_signal.models import SentimentResult
from news_signal.sentiment import HuggingFaceSentimentClassifier


class FakeTransformersPipeline:
    def __init__(self, label: str = "POSITIVE") -> None:
        self.label = label
        self.calls = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return [{"label": self.label, "score": 0.93}]


def test_classifier_returns_label_and_confidence() -> None:
    backend = FakeTransformersPipeline()
    classifier = HuggingFaceSentimentClassifier("unused", classifier=backend)

    result = classifier.classify("Profits rise", "Revenue increased.")

    assert result.label == "positive"
    assert result.confidence == 0.93
    assert backend.calls == [("Profits rise Revenue increased.", {"truncation": True})]


def test_classifier_rejects_unknown_labels() -> None:
    classifier = HuggingFaceSentimentClassifier(
        "unused", classifier=FakeTransformersPipeline("LABEL_0")
    )

    with pytest.raises(ValueError, match="unsupported sentiment label"):
        classifier.classify("Title", "Content")


def test_sentiment_confidence_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="confidence"):
        SentimentResult("positive", 1.1)
