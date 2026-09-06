import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from google.adk.sessions.state import State
from google.genai import types

from market_signal_agent.agent import build_chat_agent, build_evaluation_agent
from market_signal_agent.chat import chat, create_chat_session
from market_signal_agent.evaluation import evaluate
from market_signal_agent.models import ForecastRequest, MarketForecast
from market_signal_agent.tools import AgentTools, guard_tools


class FakeSignals:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.matches = [
            {"company": "EXAMPLE INC", "ticker": "EXM"}
        ]

    def find_companies(self, *args, **kwargs) -> list[dict[str, str]]:
        self.calls.append(("find", args, kwargs))
        return self.matches

    def get_news_signals(self, *args, **kwargs) -> dict[str, object]:
        self.calls.append(("news", args, kwargs))
        return {"signals": []}

    def get_market_signals(self, *args, **kwargs) -> dict[str, object]:
        self.calls.append(("market", args, kwargs))
        return {"signals": []}

    def get_filing_metadata(self, *args, **kwargs) -> dict[str, object]:
        self.calls.append(("filing", args, kwargs))
        return {"signals": []}


def test_company_selection_verifies_and_binds_canonical_context() -> None:
    signals = FakeSignals()
    tools = AgentTools(signals)
    context = SimpleNamespace(state=State({"company_verified": False}, {}))

    result = tools.select_company("Example", context, ticker="exm")

    assert result == {
        "status": "selected",
        "company": "EXAMPLE INC",
        "ticker": "EXM",
    }
    assert context.state["company"] == "EXAMPLE INC"
    assert context.state["ticker"] == "EXM"
    assert "news_terms" not in context.state
    assert context.state["company_verified"] is True


def test_company_selection_preserves_exchange_suffix() -> None:
    signals = FakeSignals()
    signals.matches = [
        {
            "company": "International Consolidated Airlines Group, S.A.",
            "ticker": "IAG.L",
        }
    ]
    context = SimpleNamespace(state={"company_verified": False})

    result = AgentTools(signals).select_company(
        "International Airlines Group", context, ticker="IAG.L"
    )

    assert result["ticker"] == "IAG.L"
    assert context.state["ticker"] == "IAG.L"
    assert "news_terms" not in context.state
    assert context.state["company_verified"] is True


def test_news_tool_rejects_bare_ticker_as_news_term() -> None:
    signals = FakeSignals()
    signals.matches = [
        {
            "company": "International Consolidated Airlines Group, S.A.",
            "ticker": "IAG.L",
        }
    ]
    context = SimpleNamespace(
        state={
            "company": "International Consolidated Airlines Group, S.A.",
            "ticker": "IAG.L",
            "cutoff_date": "2026-09-06T00:00:00+00:00",
            "company_verified": True,
        }
    )
    result = AgentTools(signals).get_news_signals(context, news_terms=["IAG"])

    assert result["status"] == "invalid"
    assert not signals.calls


def test_ambiguous_company_does_not_update_context() -> None:
    signals = FakeSignals()
    signals.matches = [
        {"company": "EXAMPLE INC", "ticker": "EXM"},
        {
            "company": "EXAMPLE HOLDINGS",
            "ticker": "EXH",
        },
    ]
    context = SimpleNamespace(
        state={"company": "OLD", "ticker": "OLD", "company_verified": True}
    )

    result = AgentTools(signals).select_company("Example", context)

    assert result["status"] == "ambiguous"
    assert context.state["company"] is None
    assert context.state["ticker"] is None
    assert "news_terms" not in context.state
    assert context.state["company_verified"] is False


def test_signal_guard_reports_missing_or_unverified_context() -> None:
    tool = SimpleNamespace(name="get_market_signals")
    context = SimpleNamespace(state={"benchmark": "SPY"})

    result = guard_tools(tool, {}, context)

    assert result is not None
    assert result["status"] == "blocked"
    assert result["missing"] == ["cutoff_date", "ticker", "verified company"]


def test_news_tool_uses_verified_company_and_news_terms() -> None:
    signals = FakeSignals()
    context = SimpleNamespace(
        state={
            "company": "EXAMPLE INC",
            "ticker": "EXM",
            "cutoff_date": "2024-02-20T00:00:00+00:00",
            "company_verified": True,
        }
    )

    AgentTools(signals).get_news_signals(
        context, news_terms=["Example", "Example Products"]
    )

    _, args, _ = signals.calls[-1]
    assert args[0] == "EXAMPLE INC"
    assert signals.calls[-1][2]["news_terms"] == ("Example", "Example Products")


def test_agent_configurations_expose_only_appropriate_tools() -> None:
    signals = FakeSignals()
    chat_agent = build_chat_agent("gemini-flash-latest", signals)
    evaluation_agent = build_evaluation_agent("gemini-flash-latest", signals)

    assert chat_agent.output_schema is None
    assert [tool.__name__ for tool in chat_agent.tools][:3] == [
        "select_company",
        "set_benchmark",
        "set_cutoff_date",
    ]
    assert evaluation_agent.output_schema is MarketForecast
    assert [tool.__name__ for tool in evaluation_agent.tools] == [
        "get_news_signals",
        "get_market_signals",
        "get_filing_metadata",
    ]


class FakeSessionService:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    async def create_session(self, **kwargs: object) -> object:
        self.state = kwargs["state"]
        return object()


class FakeEvent:
    def __init__(self, text: str) -> None:
        self.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

    def is_final_response(self) -> bool:
        return True


class FakeRunner:
    def __init__(self, response: str) -> None:
        self.session_service = FakeSessionService()
        self.response = response
        self.calls: list[dict[str, object]] = []

    def run_async(self, **kwargs: object):
        self.calls.append(kwargs)

        async def events():
            yield FakeEvent(self.response)

        return events()


def test_chat_reuses_the_created_session() -> None:
    runner = FakeRunner("What company should I analyse?")

    session_id = asyncio.run(create_chat_session(runner))
    result = asyncio.run(chat(runner, session_id, "Analyse a company"))

    assert result == "What company should I analyse?"
    assert runner.calls[0]["session_id"] == session_id
    assert runner.session_service.state["benchmark"] == "SPY"
    assert runner.session_service.state["company_verified"] is False


def test_evaluation_locks_context_and_validates_return_forecast() -> None:
    forecast = MarketForecast(
        ticker="EXM",
        forecast_horizon="next trading day",
        predicted_return=0.012,
        thesis="News and momentum support a modest positive return.",
        risks=["The signal may reverse."],
        evidence=[
            {
                "source": "market",
                "reference": "return_5d",
                "reason": "Shows recent momentum.",
            }
        ],
    )
    runner = FakeRunner(forecast.model_dump_json())
    request = ForecastRequest(
        company="Example Inc",
        ticker="exm",
        cutoff_date=datetime(2024, 2, 20, tzinfo=timezone.utc),
        headline="Example raises guidance",
    )

    result = asyncio.run(evaluate(runner, request))

    assert result == forecast
    assert runner.session_service.state["cutoff_date"] == "2024-02-20T00:00:00+00:00"


def test_evaluation_rejects_changed_ticker() -> None:
    runner = FakeRunner(
        MarketForecast(
            ticker="OTHER",
            forecast_horizon="next trading day",
            predicted_return=0,
            thesis="Evidence is mixed.",
            risks=[],
            evidence=[],
        ).model_dump_json()
    )
    request = ForecastRequest(
        ticker="EXM",
        cutoff_date=datetime(2024, 2, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="changed the requested ticker"):
        asyncio.run(evaluate(runner, request))
