from collections.abc import Callable
from typing import TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., object])

ARGUMENTS = {
    "company": "Company name used to search for financial news.",
    "limit": "Maximum number of articles to analyse successfully.",
    "lookback_days": "Number of days before today to search for articles.",
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
