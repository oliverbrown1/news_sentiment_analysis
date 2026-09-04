from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence

from market_signal_agent.chat import build_chat_runner, chat, create_chat_session
from news_signal_v2.config import ConfigurationError as SignalConfigurationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with the market percentage-return forecasting agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="start an interactive conversation")
    return parser


async def run_chat(
    runner,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    session_id = await create_chat_session(runner)
    while True:
        try:
            message = read("You: ").strip()
        except EOFError:
            return
        if message.casefold() in {"exit", "quit"}:
            return
        if not message:
            continue
        response = await chat(runner, session_id, message)
        write(f"Agent: {response}")


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        asyncio.run(run_chat(build_chat_runner()))
    except SignalConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"Agent error: {exc}") from exc
    return 0
