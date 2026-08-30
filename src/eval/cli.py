from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from eval.sentiment_eval import (
    SentimentLabel,
    SentimentPrediction,
    evaluate_finentity,
    load_finentity,
)
from news_signal_v1.adapters import HuggingFaceSentimentClassifier
from news_signal_v1.config import load_sentiment_model_name as load_v1_model_name
from news_signal_v2.adapters import ModernFinBertSentimentClassifier
from news_signal_v2.config import load_sentiment_model_name as load_v2_model_name
from news_signal_v2.pipeline import TargetEvidenceSelector

SystemVersion = Literal["v1", "v2"]


class V1SentimentSystem:
    def __init__(self, classifier: HuggingFaceSentimentClassifier) -> None:
        self.classifier = classifier
        self._predictions: dict[str, SentimentPrediction] = {}

    def predict(self, target: str, text: str) -> SentimentPrediction:
        del target
        if text in self._predictions:
            return self._predictions[text]
        result = self.classifier.classify("", text)
        prediction = SentimentPrediction(result.label, result.confidence)
        self._predictions[text] = prediction
        return prediction


class V2SentimentSystem:
    def __init__(self, classifier: ModernFinBertSentimentClassifier) -> None:
        self.classifier = classifier
        self._evidence_selector = TargetEvidenceSelector()
        self._predictions: dict[tuple[str, str], SentimentPrediction] = {}

    def predict(self, target: str, text: str) -> SentimentPrediction:
        key = (target, text)
        if key in self._predictions:
            return self._predictions[key]
        evidence = self._evidence_selector.select(target, None, "", text)
        result = self.classifier.classify(target, "", evidence)
        prediction = SentimentPrediction(result.label, result.confidence)
        self._predictions[key] = prediction
        return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a news signal system")
    parser.add_argument("--system", choices=("v1", "v2"), required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/finentity.json"))
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    system_version = cast(SystemVersion, args.system)
    model_name = args.model or _default_model(system_version)
    system, classifier, target_usage = _build_system(system_version, model_name)

    started = perf_counter()
    classifier.load()
    model_load_seconds = perf_counter() - started
    report = evaluate_finentity(
        system,
        load_finentity(args.dataset),
        system_name=system_version,
        model_name=model_name,
        model_revision=classifier.model_revision,
        model_load_seconds=model_load_seconds,
        target_usage=target_usage,
    )
    output = json.dumps(report.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


def _default_model(system: SystemVersion) -> str:
    return load_v1_model_name() if system == "v1" else load_v2_model_name()


def _build_system(
    system: SystemVersion, model_name: str
) -> tuple[
    V1SentimentSystem | V2SentimentSystem,
    HuggingFaceSentimentClassifier | ModernFinBertSentimentClassifier,
    str,
]:
    if system == "v1":
        classifier = HuggingFaceSentimentClassifier(model_name)
        return (
            V1SentimentSystem(classifier),
            classifier,
            "target is ignored; the full paragraph is classified once per annotation",
        )

    classifier = ModernFinBertSentimentClassifier(model_name)
    return (
        V2SentimentSystem(classifier),
        classifier,
        "target selects the relevant sentence and its immediate context",
    )
