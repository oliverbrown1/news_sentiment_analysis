from __future__ import annotations

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "market_signal_agent"
USER_ID = "market_signal_user"


def build_runner(agent: BaseAgent) -> Runner:
    return Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=InMemorySessionService(),
    )


async def run_turn(runner: Runner, session_id: str, message: str) -> str:
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )
    final_text: str | None = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            text = "".join(part.text or "" for part in event.content.parts)
            final_text = text or None
    if final_text is None:
        raise ValueError("agent did not produce a final response")
    return final_text
