import asyncio

import pytest

from market_signal_agent.entrypoints import cli


def test_cli_exposes_chat_but_not_analyse() -> None:
    assert cli.build_parser().parse_args(["chat"]).command == "chat"

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["analyse"])


def test_chat_cli_keeps_one_session_for_multiple_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = iter(["Analyse Example", "Use QQQ", "quit"])
    responses: list[str] = []
    calls: list[tuple[str, str]] = []

    async def fake_create_session(runner) -> str:
        return "session-1"

    async def fake_chat(runner, session_id: str, message: str) -> str:
        calls.append((session_id, message))
        return f"Response to {message}"

    monkeypatch.setattr(cli, "create_chat_session", fake_create_session)
    monkeypatch.setattr(cli, "chat", fake_chat)

    asyncio.run(
        cli.run_chat(
            object(),
            read=lambda prompt: next(messages),
            write=responses.append,
        )
    )

    assert calls == [
        ("session-1", "Analyse Example"),
        ("session-1", "Use QQQ"),
    ]
    assert responses == [
        "Agent: Response to Analyse Example",
        "Agent: Response to Use QQQ",
    ]
