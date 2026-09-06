from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Literal

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, ToolContext

from company_signals.entrypoints.tools import CompanySignalTools

NewsDays = Literal[7, 14, 30]
NewsLimit = Literal[5, 10, 20]
PriceDays = Literal[30, 45, 90]

SIGNAL_REQUIREMENTS = {
    "get_news_signals": ("company", "ticker", "cutoff_date"),
    "get_market_signals": ("ticker", "cutoff_date", "benchmark"),
    "get_filing_metadata": ("ticker", "cutoff_date"),
}


class AgentTools:
    def __init__(self, signals: CompanySignalTools) -> None:
        self._signals = signals

    def select_company(
        self,
        company: str,
        tool_context: ToolContext,
        ticker: str | None = None,
    ) -> dict[str, object]:
        """Resolve and select a company before collecting its signals.

        Args:
            company: Company name or ticker supplied by the user.
            ticker: Optional ticker supplied by the user for verification.
        """
        _clear_company(tool_context)
        matches = self._signals.find_companies(company, ticker)
        if not matches:
            return {
                "status": "not_found",
                "message": (
                    "No matching listed company was found; retry with a known "
                    "exchange-qualified ticker or ask the user to clarify."
                ),
            }
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "message": "Multiple companies matched; ask the user to choose one.",
                "candidates": matches,
            }

        selected = matches[0]
        tool_context.state["company"] = selected["company"]
        tool_context.state["ticker"] = selected["ticker"]
        tool_context.state["company_verified"] = True
        return {"status": "selected", **selected}

    def set_benchmark(
        self, benchmark: str, tool_context: ToolContext
    ) -> dict[str, str]:
        """Set the market benchmark used for relative performance.

        Args:
            benchmark: Benchmark ticker requested by the user.
        """
        benchmark = benchmark.strip().upper()
        if not benchmark:
            return {"status": "invalid", "message": "Benchmark cannot be empty."}
        tool_context.state["benchmark"] = benchmark
        return {"status": "selected", "benchmark": benchmark}

    def set_cutoff_date(
        self, cutoff_date: str, tool_context: ToolContext
    ) -> dict[str, str]:
        """Set the latest timestamp information may have for an analysis.

        Args:
            cutoff_date: ISO date or timezone-aware datetime requested by the user.
        """
        try:
            parsed = _parse_cutoff_date(cutoff_date)
        except ValueError as exc:
            return {"status": "invalid", "message": str(exc)}
        tool_context.state["cutoff_date"] = parsed.isoformat()
        return {"status": "selected", "cutoff_date": parsed.isoformat()}

    def get_news_signals(
        self,
        tool_context: ToolContext,
        news_days: NewsDays = 7,
        news_limit: NewsLimit = 20,
        news_terms: list[str] | None = None,
    ) -> dict[str, object]:
        """Get sentiment signals and article evidence available by the cutoff.

        Args:
            news_days: Number of days of news to consider.
            news_limit: Maximum number of successfully analysed articles.
            news_terms: Specific company or brand names to search in news titles.
        """
        try:
            terms = _normalise_news_terms(
                _state_text(tool_context, "ticker"), news_terms
            )
        except ValueError as exc:
            return {"status": "invalid", "message": str(exc)}
        return self._signals.get_news_signals(
            _state_text(tool_context, "company"),
            _state_text(tool_context, "ticker"),
            _state_date(tool_context),
            news_days=news_days,
            news_limit=news_limit,
            news_terms=terms,
        )

    def get_market_signals(
        self,
        tool_context: ToolContext,
        price_days: PriceDays = 45,
    ) -> dict[str, object]:
        """Get price, activity, and benchmark signals available by the cutoff.

        Args:
            price_days: Number of calendar days of price history to consider.
        """
        return self._signals.get_market_signals(
            _state_text(tool_context, "ticker"),
            _state_date(tool_context),
            benchmark=_state_text(tool_context, "benchmark"),
            price_days=price_days,
        )

    def get_filing_metadata(self, tool_context: ToolContext) -> dict[str, object]:
        """Get the latest supported SEC filing metadata available by the cutoff."""
        return self._signals.get_filing_metadata(
            _state_text(tool_context, "ticker"), _state_date(tool_context)
        )


def guard_tools(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: CallbackContext,
) -> dict[str, object] | None:
    del args
    required = SIGNAL_REQUIREMENTS.get(tool.name)
    if required is None:
        return None
    missing = [name for name in required if not tool_context.state.get(name)]
    if not tool_context.state.get("company_verified"):
        missing.append("verified company")
    if missing:
        return {
            "status": "blocked",
            "message": "Required analysis context is missing.",
            "missing": sorted(set(missing)),
        }
    return None


def _state_text(tool_context: ToolContext, key: str) -> str:
    value = tool_context.state.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing trusted agent state: {key}")
    return value


def _state_date(tool_context: ToolContext) -> datetime:
    return datetime.fromisoformat(_state_text(tool_context, "cutoff_date"))


def _clear_company(tool_context: ToolContext) -> None:
    tool_context.state.update(
        {
            "company": None,
            "ticker": None,
            "company_verified": False,
        }
    )


def _normalise_news_terms(
    ticker: str,
    supplied: list[str] | None,
) -> tuple[str, ...] | None:
    if supplied is None:
        return None
    terms: list[str] = []
    seen: set[str] = set()
    bare_ticker = ticker.split(".", maxsplit=1)[0].casefold()
    for value in supplied:
        term = value.strip()
        if not 2 <= len(term) <= 80:
            raise ValueError("each news term must contain between 2 and 80 characters")
        if '"' in term:
            raise ValueError("news terms cannot contain quotes")
        if term.casefold() in {ticker.casefold(), bare_ticker}:
            raise ValueError("news terms cannot be the selected ticker alone")
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)
    if len(terms) > 5:
        raise ValueError("at most five news terms may be selected")
    if not terms:
        raise ValueError("at least one news term must be supplied")
    return tuple(terms)


def _parse_cutoff_date(value: str) -> datetime:
    try:
        if "T" not in value:
            return datetime.combine(date.fromisoformat(value), time.min, timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("cutoff date must be an ISO date or datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("cutoff datetime must include a timezone")
    return parsed.astimezone(timezone.utc)
