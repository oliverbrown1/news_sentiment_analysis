from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from company_signals.providers import YFinancePriceProvider
from eval.arguments import SEED_ARGUMENTS
from eval.market_dataset import (
    build_market_dataset,
    load_market_seed,
    write_market_dataset,
)
from eval.models import EvaluationDataError
from news_signal_v2.adapters import NewsApiProvider
from news_signal_v2.config import ConfigurationError
from news_signal_v2.config import Settings

DEFAULT_SEED = Path("data/market_eval_seed.json")
DEFAULT_OUTPUT = Path("data/recent_market_headlines.jsonl")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a recent headline-level market evaluation dataset"
    )
    parser.add_argument(
        "--seed", type=Path, default=DEFAULT_SEED, help=SEED_ARGUMENTS["seed"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=SEED_ARGUMENTS["seed_output"],
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=SEED_ARGUMENTS["execute"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=SEED_ARGUMENTS["overwrite"],
    )
    return parser

# Seed a recent headline dataset with the following trading day's market reaction.
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_file = load_market_seed(args.seed)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "seed": seed_file.path,
                    "output": str(args.output),
                    "seed_windows": len(seed_file.seeds),
                    "maximum_headlines": sum(
                        seed.article_limit for seed in seed_file.seeds
                    ),
                    "maximum_newsapi_requests": len(seed_file.seeds) * 2,
                    "message": "Pass --execute to perform external requests.",
                },
                indent=2,
            )
        )
        return 0
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}; pass --overwrite to replace it"
        )

    try:
        settings = Settings.from_env()
        dataset = build_market_dataset(
            seed_file,
            NewsApiProvider(
                settings.news_api_key,
                api_url=settings.news_api_url,
                domains=settings.news_domains,
            ),
            YFinancePriceProvider(),
        )
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except EvaluationDataError as exc:
        raise SystemExit(f"Dataset build failed: {exc}") from exc
    for failure in dataset.build_failures:
        LOGGER.warning("Seed %s failed: %s", failure.seed_id, failure.reason)
    write_market_dataset(dataset, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "examples": len(dataset.examples),
                "failures": len(dataset.build_failures),
                "build_failures": [
                    {"seed_id": failure.seed_id, "reason": failure.reason}
                    for failure in dataset.build_failures
                ],
            },
            indent=2,
        )
    )
    return 0
