import argparse
from collections.abc import Callable
from datetime import date, datetime, time, timezone
from typing import TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., object])

ARGUMENTS = {
    "company": "Company name used to search for news and label the result.",
    "ticker": "Listed symbol, including its exchange suffix when required.",
    "cutoff_date": "Latest timestamp data may have for point-in-time analysis.",
    "benchmark": "Market ticker used to compare the company's price performance.",
    "news_days": "Number of days before the cutoff to search for news.",
    "news_limit": "Maximum number of articles to analyse successfully.",
    "news_terms": "Specific company or brand names to search in news titles.",
    "price_days": "Number of calendar days of price history to request.",
    "verbose": "Include article excerpts and individual failure details.",
}


def documented(
    summary: str, arguments: tuple[str, ...]
) -> Callable[[CallableType], CallableType]:
    def decorate(function: CallableType) -> CallableType:
        parameters = "\n".join(
            f"    {name}: {ARGUMENTS[name]}" for name in arguments
        )
        function.__doc__ = f"{summary}\n\nArgs:\n{parameters}"
        return function

    return decorate


def parse_cutoff_date(value: str) -> datetime:
    try:
        if "T" not in value:
            return datetime.combine(date.fromisoformat(value), time.min, timezone.utc)
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cutoff-date must be an ISO date or datetime"
        ) from exc
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "cutoff-date datetime must include a timezone"
        )
    return result.astimezone(timezone.utc)
