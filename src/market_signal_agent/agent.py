from __future__ import annotations

from google.adk.agents import LlmAgent
from google.genai import types

from company_signals.entrypoints.tools import CompanySignalTools
from market_signal_agent.models import MarketForecast
from market_signal_agent.tools import AgentTools, guard_tools

FORECAST_INSTRUCTION = """Use point-in-time company signals to predict the ticker's percentage
return over the next trading day. Express predicted_return as a decimal, where 0.012 means 1.2%.
Sentiment is evidence, not the final prediction. Prefer a return near zero when evidence is weak
or contradictory.
Every material claim must cite a headline, signal name, article URL, or filing URL. Never invent
values, filing contents, or references, and never use information after the cutoff date. Filing
metadata proves only that a form was filed at a given time; it does not reveal the filing's contents.
SEC filing coverage is optional and may be unavailable for non-US-listed companies; this does not
invalidate their Yahoo Finance price signals or news evidence.
Treat news coverage as retrieval diagnostics: zero retrieved articles means no evidence was returned
for that query and lookback, not that the company had no news or catalysts. Use the search strategy
and retrieved, attempted, relevant, and analysed counts when judging how much weight to give news.
Relevant means target evidence was found, not that the article is financially material.
"""

CHAT_INSTRUCTION = FORECAST_INSTRUCTION + """

You are conversational. Resolve the company with select_company before analysis, preserving exchange
suffixes such as .L. If a name-only lookup fails and you know its exchange-qualified ticker, retry
select_company with both values; ticker validation will reject a mismatch. Supply up to five specific
issuer, product, or brand names to get_news_signals as news_terms when they improve news recall. Do
not use broad sector terms, a ticker alone, or an ambiguous geographic or common term without a
company qualifier. Ask the user to choose when multiple candidates remain. The benchmark defaults
to SPY and the cutoff defaults to now; change either only when the user requests it. If a signal tool
is blocked, obtain the missing context rather than guessing. Always inspect news and market signals;
inspect filing metadata when useful. Once analysis is complete, report the same fields as
MarketForecast: ticker, forecast_horizon, predicted_return, thesis, risks, and evidence.
"""

EVALUATION_INSTRUCTION = FORECAST_INSTRUCTION + """

The company, ticker, benchmark, cutoff date, headline, and forecast horizon are fixed in session
state. The supplied headline is the complete news input: do not retrieve more news. Inspect market
signals and filing metadata when useful, then return only the required MarketForecast.
"""


def build_chat_agent(model: str, signal_tools: CompanySignalTools) -> LlmAgent:
    tools = AgentTools(signal_tools)
    return LlmAgent(
        name="market_signal_chat",
        description="Discusses companies and predicts next-day percentage returns.",
        model=model,
        instruction=CHAT_INSTRUCTION,
        tools=[
            tools.select_company,
            tools.set_benchmark,
            tools.set_cutoff_date,
            tools.get_news_signals,
            tools.get_market_signals,
            tools.get_filing_metadata,
        ],
        before_tool_callback=guard_tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )


def build_evaluation_agent(model: str, signal_tools: CompanySignalTools) -> LlmAgent:
    tools = AgentTools(signal_tools)
    return LlmAgent(
        name="market_signal_evaluation",
        description="Predicts next-day percentage returns from locked point-in-time context.",
        model=model,
        instruction=EVALUATION_INSTRUCTION,
        tools=[
            tools.get_market_signals,
            tools.get_filing_metadata,
        ],
        before_tool_callback=guard_tools,
        output_schema=MarketForecast,
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )
