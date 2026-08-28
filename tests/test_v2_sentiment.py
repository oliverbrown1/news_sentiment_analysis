import pytest

from news_signal_v2.adapters import ModernFinBertSentimentClassifier
from news_signal_v2.models import SentimentClassificationError


class FakeBackend:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return [self.output]


def test_v2_classifier_returns_normalized_result() -> None:
    backend = FakeBackend({"label": "POSITIVE", "score": 0.94})
    classifier = ModernFinBertSentimentClassifier("test/model", backend)

    result = classifier.classify("Example Ltd", "Profit rises", "Revenue increased")

    assert result.label == "positive"
    assert result.confidence == 0.94
    assert backend.calls[0][1] == {"truncation": True}


def test_v2_classifier_rejects_unknown_label() -> None:
    classifier = ModernFinBertSentimentClassifier(
        "test/model", FakeBackend({"label": "BULLISH", "score": 0.9})
    )

    with pytest.raises(SentimentClassificationError, match="unsupported"):
        classifier.classify("Example Ltd", "Title", "Content")


@pytest.mark.parametrize(
    "raw_label, expected",
    [("LABEL_0", "negative"), ("LABEL_1", "neutral"), ("LABEL_2", "positive")],
)
def test_v2_classifier_maps_checkpoint_label_ids(
    raw_label: str, expected: str
) -> None:
    classifier = ModernFinBertSentimentClassifier(
        "test/model", FakeBackend({"label": raw_label, "score": 0.9})
    )

    assert classifier.classify("Example Ltd", "Title", "Content").label == expected
