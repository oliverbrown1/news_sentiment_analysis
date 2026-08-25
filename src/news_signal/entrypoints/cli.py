from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from news_signal.adapters import HuggingFaceSentimentClassifier
from news_signal.application import build_pipeline
from news_signal.config import ConfigurationError, Settings, load_sentiment_model_name
from news_signal.evaluation import EvaluationDataError, evaluate_finentity, load_finentity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse recent financial news sentiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="analyse news about a company")
    analyse.add_argument("--company", required=True)
    analyse.add_argument("--limit", type=int, default=5)
    analyse.add_argument("--days", type=int)

    evaluate = subparsers.add_parser(
        "evaluate-sentiment", help="evaluate sentiment classification on FinEntity"
    )
    evaluate.add_argument("--dataset", type=Path, default=Path("data/finentity.json"))
    evaluate.add_argument("--model")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate-sentiment":
        return _evaluate_sentiment(args)

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    lookback_days = args.days if args.days is not None else settings.lookback_days
    result = build_pipeline(settings).analyse(args.company, args.limit, lookback_days)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0

# Uses FinEntity dataset
def _evaluate_sentiment(args: argparse.Namespace) -> int:
    try:
        model_name = args.model or load_sentiment_model_name()
        dataset = load_finentity(args.dataset)
    except (ConfigurationError, EvaluationDataError) as exc:
        raise SystemExit(f"Evaluation error: {exc}") from exc

    classifier = HuggingFaceSentimentClassifier(model_name)
    started = perf_counter()
    classifier.load()
    model_load_seconds = perf_counter() - started
    report = evaluate_finentity(
        classifier,
        dataset,
        model_name=model_name,
        model_revision=classifier.model_revision,
        model_load_seconds=model_load_seconds,
    )
    output = json.dumps(report.to_dict(), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
    print(output)
    return 0
