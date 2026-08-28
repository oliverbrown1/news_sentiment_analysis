from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from news_signal_v2.application import build_pipeline
from news_signal_v2.config import ConfigurationError, Settings
from news_signal_v2.models import NewsProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse financial news with V2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyse = subparsers.add_parser("analyse", help="analyse news about a company")
    analyse.add_argument("--company", required=True)
    analyse.add_argument("--ticker")
    analyse.add_argument("--limit", type=int, default=5)
    analyse.add_argument("--days", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        lookback_days = args.days if args.days is not None else settings.lookback_days
        result = build_pipeline(settings).analyse(
            company=args.company,
            ticker=args.ticker,
            limit=args.limit,
            lookback_days=lookback_days,
        )
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except NewsProviderError as exc:
        raise SystemExit(f"News provider error: {exc}") from exc

    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0
