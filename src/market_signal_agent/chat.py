from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from google.adk.runners import Runner

from company_signals.entrypoints.tools import build_tools
from market_signal_agent.agent import build_chat_agent
from market_signal_agent.config import get_model
from market_signal_agent.runner import APP_NAME, USER_ID, build_runner, run_turn


def build_chat_runner(model: str | None = None) -> Runner:
    return build_runner(build_chat_agent(model or get_model(), build_tools()))


async def create_chat_session(runner: Runner) -> str:
    session_id = uuid4().hex
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            "benchmark": "SPY",
            "cutoff_date": datetime.now(timezone.utc).isoformat(),
            "company_verified": False,
        },
    )
    return session_id


async def chat(runner: Runner, session_id: str, message: str) -> str:
    if not message.strip():
        raise ValueError("message cannot be empty")
    return await run_turn(runner, session_id, message.strip())
