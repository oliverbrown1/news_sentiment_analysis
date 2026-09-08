from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from eval.agent_eval import AdkAgentSystem, AgentEvaluationReport, evaluate_market_agent
from eval.arguments import ARGUMENTS
from eval.market_dataset import load_market_dataset
from eval.market_eval import MarketEvaluationReport, evaluate_finmarba, load_finmarba
from eval.sentiment_baseline import SentimentMarketBaseline
from eval.sentiment_eval import (
    SentimentEvaluationReport,
    SentimentPrediction,
    evaluate_finentity,
    load_finentity,
)
from news_signal_v1.adapters import HuggingFaceSentimentClassifier
from news_signal_v1.config import load_sentiment_model_name as load_v1_model_name
from news_signal_v2.adapters import ModernFinBertSentimentClassifier
from news_signal_v2.config import load_sentiment_model_name as load_v2_model_name
from news_signal_v2.pipeline import TargetEvidenceSelector
from market_signal_agent.config import get_model as get_agent_model

SystemVersion = Literal["v1", "v2", "agent"]
TaskName = Literal["sentiment", "market", "agent"]


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
    parser.add_argument(
        "--task",
        choices=("sentiment", "market", "agent"),
        default="sentiment",
        help=ARGUMENTS["task"],
    )
    parser.add_argument(
        "--system",
        choices=("v1", "v2", "agent"),
        required=True,
        help=ARGUMENTS["system"],
    )
    parser.add_argument("--dataset", type=Path, help=ARGUMENTS["dataset"])
    parser.add_argument("--model", help=ARGUMENTS["model"])
    parser.add_argument("--output", type=Path, help=ARGUMENTS["output"])
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, help=ARGUMENTS["limit"])
    scope.add_argument("--all", action="store_true", help=ARGUMENTS["all"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = cast(TaskName, args.task)
    system_version = cast(SystemVersion, args.system)
    if task == "market" and system_version != "v2":
        raise SystemExit("Market evaluation currently supports only --system v2")
    if task == "agent" and system_version != "agent":
        raise SystemExit("Agent evaluation requires --system agent")
    if task == "sentiment" and system_version == "agent":
        raise SystemExit("Sentiment evaluation supports only --system v1 or v2")
    if (args.limit is not None or args.all) and task != "agent":
        raise SystemExit("--limit and --all are supported only for agent evaluation")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    model_name = args.model or (
        get_agent_model() if system_version == "agent" else _default_model(system_version)
    )
    if task == "agent":
        dataset_path = args.dataset or Path("data/recent_market_headlines.jsonl")
        limit = None if args.all else args.limit or 8
        report = _evaluate_agent(model_name, dataset_path, limit)
    elif task == "market":
        dataset_path = args.dataset or Path("data/finmarba.csv")
        report = _evaluate_market(model_name, dataset_path)
    else:
        dataset_path = args.dataset or Path("data/finentity.json")
        report = _evaluate_sentiment(system_version, model_name, dataset_path)

    output = json.dumps(report.to_dict(), indent=2)
    output_path = args.output or _default_output_path(
        task, system_version, full_agent_run=args.all
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)

    return 0


def _default_output_path(
    task: TaskName,
    system: SystemVersion,
    *,
    full_agent_run: bool,
) -> Path:
    if task == "sentiment":
        filename = f"finentity-sentiment-{system}.json"
    elif task == "market":
        filename = f"finmarba-market-sentiment-{system}.json"
    else:
        filename = (
            "market-agent-headline-v1.json"
            if full_agent_run
            else "market-agent-headline-review-v1.json"
        )
    return Path("reports") / filename


def _evaluate_sentiment(
    system_version: SystemVersion, model_name: str, dataset_path: Path
) -> SentimentEvaluationReport:
    system, classifier, target_usage = _build_system(system_version, model_name)

    started = perf_counter()
    classifier.load()
    model_load_seconds = perf_counter() - started
    report = evaluate_finentity(
        system,
        load_finentity(dataset_path),
        system_name=system_version,
        model_name=model_name,
        model_revision=classifier.model_revision,
        model_load_seconds=model_load_seconds,
        target_usage=target_usage,
    )
    return report


def _evaluate_market(model_name: str, dataset_path: Path) -> MarketEvaluationReport:
    baseline, classifier = _build_market_baseline(model_name)
    started = perf_counter()
    classifier.load()
    model_load_seconds = perf_counter() - started
    return evaluate_finmarba(
        baseline,
        load_finmarba(dataset_path),
        system_name="sentiment-v2-baseline",
        model_name=model_name,
        model_revision=classifier.model_revision,
        model_load_seconds=model_load_seconds,
    )


def _evaluate_agent(
    model_name: str, dataset_path: Path, limit: int | None
) -> AgentEvaluationReport:
    return asyncio.run(
        evaluate_market_agent(
            AdkAgentSystem(model_name),
            load_market_dataset(dataset_path),
            system_name="market-signal-agent",
            model_name=model_name,
            limit=limit,
        )
    )


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


def _build_market_baseline(
    model_name: str,
) -> tuple[SentimentMarketBaseline, ModernFinBertSentimentClassifier]:
    classifier = ModernFinBertSentimentClassifier(model_name)
    return SentimentMarketBaseline(classifier), classifier
