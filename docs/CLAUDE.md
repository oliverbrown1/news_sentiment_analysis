# News Sentiment Analysis: Project Guide

Last reviewed: 2026-08-30

## Purpose

This repository contains comparable V1 and V2 financial-news pipelines plus an
an independent evaluation package. V1 preserves the modernised legacy behavior.
V2 adds broader ingestion, stronger article extraction, company-specific
evidence, and an updated sentiment model.

Neither version predicts company or share-price performance. Keep that claim out
of documentation until a leakage-safe backtest exists.

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
tests/                     # offline pytest suite for all three packages
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
direct NewsAPI request + optional domains + optional ticker
  -> canonical URL/title deduplication
  -> Trafilatura article body extraction
  -> target sentence and immediate-context selection
  -> ModernFinBERT sentiment
  -> JSON or agent-tool dictionary
```

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
`MarketSystem.predict(as_of, ticker, headline)`. It reuses the same metrics but
compares predictions with market-derived direction rather than semantic
sentiment. The public subset has 8,142 rows covering 2010-2011 and yields 9,978
labelled ticker examples; it is not the full corpus described by the paper.

## Commands

```bash
uv sync
cp .env.example .env
uv run python -m nltk.downloader punkt punkt_tab

uv run news-signal-v1 analyse --company "NVIDIA" --limit 5
uv run news-signal-v2 analyse --company "NVIDIA" --ticker NVDA --limit 5

uv run news-signal-eval --system v1 --output reports/finentity-sentiment-v1.json
uv run news-signal-eval --system v2 --output reports/finentity-sentiment-v2.json
uv run pytest
```

`news-signal` aliases V2. `NewsSentimentAnalysis.py` remains a V1 compatibility
wrapper. Use `uv add`, `uv remove`, `uv lock`, and `uv build`; do not add pip
requirements files alongside `pyproject.toml` and `uv.lock`.

## Configuration

- `NEWS_API_KEY`: required for live news analysis, never for evaluation.
- `NEWS_LOOKBACK_DAYS`: positive integer, default `7`.
- `SENTIMENT_MODEL`: optional V1 Hugging Face model identifier.
- `V2_SENTIMENT_MODEL`: optional V2 Hugging Face model identifier.
- `NEWS_DOMAINS`: optional comma-separated V2 source filter; empty means no filter.
- `NEWS_API_URL`: optional V2 endpoint, useful for controlled integration tests.

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

## Current Limitations

- Processing is sequential and has no retry, cache, persistence, or tracing.
- NewsAPI coverage and publisher extraction remain externally constrained.
- Target evidence selection is a deterministic baseline.
- FinEntity evaluates sentiment only; FinMarBa evaluates market direction only.
- No production system implements the market prediction contract yet.
- There is no HTTP API, signal aggregation, deployment, or market backtest yet.

Continue with one numbered step from `docs/PLAN.MD` at a time. Steps 1-4 are
implemented; Step 5 is the next application change.
