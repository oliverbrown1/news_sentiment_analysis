from __future__ import annotations

from uuid import uuid4

from google.adk.runners import Runner

from company_signals.entrypoints.tools import build_tools
from market_signal_agent.agent import build_evaluation_agent
from market_signal_agent.config import get_model
from market_signal_agent.models import ForecastRequest, MarketForecast
from market_signal_agent.runner import APP_NAME, USER_ID, build_runner, run_turn


def build_evaluation_runner(model: str | None = None) -> Runner:
    return build_runner(build_evaluation_agent(model or get_model(), build_tools()))


async def evaluate(runner: Runner, request: ForecastRequest) -> MarketForecast:
    session_id = uuid4().hex
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            "company": request.company or request.ticker,
            "ticker": request.ticker,
            "cutoff_date": request.cutoff_date.isoformat(),
            "benchmark": request.benchmark,
            "headline": request.headline,
            "forecast_horizon": request.forecast_horizon,
            "company_verified": True,
        },
    )
    final_text = await run_turn(runner, session_id, _prompt(request))
    try:
        result = MarketForecast.model_validate_json(final_text)
    except ValueError as exc:
        raise ValueError("agent returned an invalid market forecast") from exc
    if result.ticker != request.ticker:
        raise ValueError("agent changed the requested ticker")
    if result.forecast_horizon != request.forecast_horizon:
        raise ValueError("agent changed the forecast horizon")
    return result


def _prompt(request: ForecastRequest) -> str:
    headline = request.headline or "No specific headline was supplied."
    company = request.company or request.ticker
    return (
        f"Forecast {company} ({request.ticker}). Cutoff: "
        f"{request.cutoff_date.isoformat()}. Horizon: {request.forecast_horizon}. "
        f"Headline: {headline}"
    )
