from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from news_signal_v1.application import build_pipeline
from news_signal_v1.config import ConfigurationError, Settings
from news_signal_v1.entrypoints.arguments import ARGUMENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse recent financial news sentiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="analyse news about a company")
    analyse.add_argument("--company", required=True, help=ARGUMENTS["company"])
    analyse.add_argument("--limit", type=int, default=5, help=ARGUMENTS["limit"])
    analyse.add_argument("--days", type=int, help=ARGUMENTS["lookback_days"])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    lookback_days = args.days if args.days is not None else settings.lookback_days
    result = build_pipeline(settings).analyse(args.company, args.limit, lookback_days)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0
