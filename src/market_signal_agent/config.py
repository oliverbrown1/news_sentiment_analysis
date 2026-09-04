from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-flash-latest"


def get_model() -> str:
    load_dotenv(Path.cwd() / ".env")
    return os.getenv("MARKET_SIGNAL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
