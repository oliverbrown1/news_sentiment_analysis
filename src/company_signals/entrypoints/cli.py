from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from company_signals.application import build_pipeline
from company_signals.entrypoints.arguments import ARGUMENTS, parse_cutoff_date
from news_signal_v2.config import ConfigurationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect point-in-time company signals")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect signals for a company")
    collect.add_argument(
        "--company", required=True, help=ARGUMENTS["company"]
    )
    collect.add_argument(
        "--ticker", required=True, help=ARGUMENTS["ticker"]
    )
    collect.add_argument(
        "--cutoff-date",
        required=True,
        type=parse_cutoff_date,
        help=ARGUMENTS["cutoff_date"],
    )
    collect.add_argument(
        "--benchmark", default="SPY", help=ARGUMENTS["benchmark"]
    )
    collect.add_argument(
        "--news-days", type=int, default=7, help=ARGUMENTS["news_days"]
    )
    collect.add_argument(
        "--news-limit",
        type=int,
        default=20,
        help=ARGUMENTS["news_limit"],
    )
    collect.add_argument(
        "--price-days",
        type=int,
        default=45,
        help=ARGUMENTS["price_days"],
    )
    collect.add_argument(
        "--verbose", action="store_true", help=ARGUMENTS["verbose"]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_pipeline().collect(
            company=args.company,
            ticker=args.ticker,
            cutoff_date=args.cutoff_date,
            benchmark=args.benchmark,
            news_days=args.news_days,
            news_limit=args.news_limit,
            price_days=args.price_days,
        )
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    print(json.dumps(result.to_dict(verbose=args.verbose), indent=2, default=str))
    return 0
