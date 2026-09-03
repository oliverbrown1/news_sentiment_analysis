from collections.abc import Callable
from typing import TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., object])

ARGUMENTS = {
    "company": "Company name used to search for financial news.",
    "ticker": "Optional stock-market symbol used to improve the news search.",
    "limit": "Maximum number of articles to analyse successfully.",
    "lookback_days": "Number of days before the analysis endpoint to search for articles.",
    "cutoff_date": "Latest timestamp an article may have; omit to use the current time.",
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
