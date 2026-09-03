from collections.abc import Callable
from typing import TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., object])

ARGUMENTS = {
    "company": "Company name used to search for news and label the result.",
    "ticker": "Stock-market symbol used to retrieve prices and SEC filings.",
    "cutoff_date": "Latest timestamp data may have for point-in-time analysis.",
    "benchmark": "Market ticker used to compare the company's price performance.",
    "news_days": "Number of days before the cutoff to search for news.",
    "news_limit": "Maximum number of articles to analyse successfully.",
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
