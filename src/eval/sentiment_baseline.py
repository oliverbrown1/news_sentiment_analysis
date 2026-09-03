from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from eval.market_eval import MarketPrediction
from news_signal_v2.adapters import ModernFinBertSentimentClassifier

SCORES = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


class SentimentMarketBaseline:
    def __init__(self, classifier: ModernFinBertSentimentClassifier) -> None:
        self.classifier = classifier
        self._predictions: dict[str, MarketPrediction] = {}

    def predict(
        self, cutoff_date: date, ticker: str, headline: str
    ) -> MarketPrediction:
        del cutoff_date
        if headline in self._predictions:
            return self._predictions[headline]

        sentiment = self.classifier.classify(ticker, "", headline)
        prediction = MarketPrediction(sentiment.label, sentiment.confidence)
        self._predictions[headline] = prediction
        return prediction


def aggregate_sentiment(
    predictions: Sequence[MarketPrediction], neutral_threshold: float = 0.2
) -> MarketPrediction:
    if not predictions:
        raise ValueError("at least one sentiment prediction is required")
    if not 0 <= neutral_threshold < 1:
        raise ValueError("neutral_threshold must be between 0 and 1")

    confidence_total = sum(item.confidence for item in predictions)
    if confidence_total == 0:
        return MarketPrediction("neutral", 0.0)

    score = sum(
        SCORES[item.label] * item.confidence for item in predictions
    ) / confidence_total
    if score > neutral_threshold:
        return MarketPrediction("positive", abs(score))
    if score < -neutral_threshold:
        return MarketPrediction("negative", abs(score))
    return MarketPrediction("neutral", 1 - abs(score))
