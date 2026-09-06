# News Sentiment Analysis: Project Guide

Last reviewed: 2026-09-06

## Purpose

This repository contains comparable V1 and V2 financial-news pipelines plus an
independent evaluation package and a Google ADK market forecasting agent. V1 preserves the modernised legacy behavior.
V2 adds broader ingestion, stronger article extraction, company-specific
evidence, and an updated sentiment model.

The agent produces an unvalidated next-trading-day return forecast. Do not make
predictive-performance claims until leakage-safe evaluation and backtesting exist.

## Repository Layout

```text
src/
  eval/
    sentiment_eval.py      # FinEntity semantic-sentiment evaluation
    market_eval.py         # FinMarBa market-direction evaluation
    metrics.py             # shared classification, calibration and latency metrics
    models.py              # shared labels and prediction types
  news_signal_v1/          # preserved NewsAPI/newspaper3k/DistilRoBERTa baseline
    entrypoints/           # V1 CLI and reusable agent tool
  news_signal_v2/          # improved ingestion and analysis pipeline
    entrypoints/           # V2 CLI and reusable agent tool
  company_signals/         # point-in-time news, market and filing signals
    entrypoints/           # signal CLI, tool and shared argument descriptions
  market_signal_agent/     # conversational and locked-evaluation ADK runtimes
tests/                     # offline pytest suite for all packages
data/finentity.json        # pinned entity-level sentiment benchmark
data/finmarba.csv          # pinned released market-direction subset
reports/                   # generated evaluation reports
```

Each news package keeps `models.py` for immutable result types, `config.py` for
environment configuration, `adapters.py` for external integrations,
`pipeline.py` for orchestration and injectable protocols, and `application.py`
for production construction. Transport code remains in `entrypoints/`.

## Runtime Flows

V1:

```text
NewsAPI SDK + fixed domain allowlist
  -> newspaper3k article summary
  -> legacy financial DistilRoBERTa sentiment
  -> JSON or agent-tool dictionary
```

V2:

```text
verified common company alias -> title-only NewsAPI search in configured domains
  -> unrestricted title-search fallback when fewer than five candidates are returned
  -> canonical URL/title deduplication
  -> Trafilatura article body extraction
  -> target sentence and immediate-context selection
  -> ModernFinBERT sentiment
  -> JSON or agent-tool dictionary
```

Company signals:

```text
dated V2 news analysis + global Yahoo Finance OHLCV + optional dated SEC filings
  -> deterministic news, momentum, activity, relative and filing signals
  -> timestamped values, failures and cited source evidence
  -> JSON or agent-tool dictionary
```

Market signal agent:

```text
free-text chat -> verified Yahoo Finance identity + agent-selected news terms
  -> guarded signal tools
  -> next-trading-day percentage-return forecast with cited evidence
```

The evaluation runtime instead receives fixed company, ticker, benchmark and
cutoff state, omits all context-changing tools and returns `MarketForecast` JSON.

## Evaluation

`eval.sentiment_eval` owns FinEntity parsing and depends on the small
`SentimentSystem.predict(target, text)` protocol. The CLI adapters make the
systems' different behavior explicit:

- V1 classifies the whole paragraph and ignores the entity target.
- V2 uses the entity target to select evidence before classification.

Reports contain accuracy, macro F1, per-class precision/recall/F1, confusion
matrix, calibration, per-annotation latency, dataset diagnostics, a mixed-label
slice, and high-confidence errors. Dataset revision and SHA-256, model revision,
runtime versions, failures, and target usage are recorded for reproducibility.

`eval.market_eval` owns the pinned released FinMarBa subset and defines
`MarketSystem.predict(cutoff_date, ticker, headline)`. It reuses the same metrics but
compares predictions with market-derived direction rather than semantic
sentiment. The public subset has 8,142 rows covering 2010-2011 and yields 9,978
labelled ticker examples; it is not the full corpus described by the paper.

`eval.sentiment_baseline` implements the first market system by passing each
headline through the V2 sentiment classifier and reusing its label and confidence
as market direction. It deliberately uses no price, date, fundamental, or agent
inputs. The pinned report records 46.67% accuracy, 45.66% macro F1, and 50.00%
expected calibration error.

## Commands

```bash
uv sync
cp .env.example .env
uv run python -m nltk.downloader punkt punkt_tab

uv run news-signal-v1 analyse --company "NVIDIA" --limit 5
uv run news-signal-v2 analyse --company "NVIDIA" --ticker NVDA --limit 5
uv run company-signals collect --company "NVIDIA" --ticker NVDA --cutoff-date 2026-08-30
uv run company-signals collect --company "International Airlines Group" --ticker IAG.L --cutoff-date 2026-09-06 --news-term "British Airways" --news-term "Iberia"
uv run market-signal-agent chat

uv run news-signal-eval --system v1 --output reports/finentity-sentiment-v1.json
uv run news-signal-eval --system v2 --output reports/finentity-sentiment-v2.json
uv run news-signal-eval --task market --system v2 --output reports/finmarba-market-sentiment-v2.json
uv run pytest
```

`news-signal` aliases V2. `NewsSentimentAnalysis.py` remains a V1 compatibility
wrapper. Use `uv add`, `uv remove`, `uv lock`, and `uv build`; do not add pip
requirements files alongside `pyproject.toml` and `uv.lock`.

Company signal output is compact by default: it reports article coverage,
sentiment-bearing references, and grouped failures. Add `--verbose` for full
evidence excerpts and individual failure details.

News coverage reports the quoted query and search strategy plus `retrieved`,
`attempted`, `relevant`, and `analysed` counts. Zero retrieved articles means
the query returned no evidence; it does not prove that no company news existed.
`Relevant` means the selector found the company or ticker, not that the article
was financially material.

Company identity and news discovery are separate: Yahoo Finance verifies the
company and exchange-qualified ticker, while each news request may provide up
to five specific company or brand names as `news_terms`. V2 combines those
terms into one title query; broad sector terms and bare ticker terms are not
allowed. News terms are request parameters and are not stored in agent state.

## Configuration

- `NEWS_API_KEY`: required for live news analysis, never for evaluation.
- `NEWS_LOOKBACK_DAYS`: positive integer, default `7`.
- `SENTIMENT_MODEL`: optional V1 Hugging Face model identifier.
- `V2_SENTIMENT_MODEL`: optional V2 Hugging Face model identifier.
- `NEWS_DOMAINS`: optional comma-separated V2 source filter; empty means no filter.
- `NEWS_API_URL`: optional V2 endpoint, useful for controlled integration tests.
- `SEC_USER_AGENT`: required for optional SEC filing lookup and should include a contact email.
- `GOOGLE_API_KEY`: used by Google ADK for Gemini requests.
- `MARKET_SIGNAL_MODEL`: optional Gemini model, default `gemini-flash-latest`.

The first real inference can download a Hugging Face model. Imports and unit
tests must stay offline and side-effect free. Never commit `.env`, credentials,
downloaded articles, or model weights.

## Development Rules

- Keep external services behind the existing protocols and inject fakes in tests.
- Preserve URLs, timestamps, confidence, evidence, and explicit failure details.
- Do not catch broad exceptions or turn failures into successful empty results.
- Keep deterministic scoring separate from model-generated output.
- Keep result models typed and serialization at entry-point boundaries.
- Evaluate changes before making quality or predictive claims.
- Remove a configured news domain only after extraction fails for at least three
  distinct URLs across at least two company queries.

## Current Limitations

- Processing is sequential and has no retry, cache, persistence, or tracing.
- NewsAPI coverage and publisher extraction remain externally constrained.
- Target evidence selection is a deterministic baseline.
- FinEntity evaluates sentiment; the current FinMarBa evaluator still evaluates direction.
- Agent return regression and derived FinMarBa-class metrics remain Step 8 work.
- There is no HTTP API, deployment, market backtest, cache, persistence, or tracing yet.

Continue with one numbered step from `docs/PLAN.MD` at a time. Steps 1-6 are
implemented; Step 8 is the next evaluation change.
