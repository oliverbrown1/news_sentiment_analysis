# News Sentiment Analysis: Project Guide

Last reviewed: 2026-08-24

## Purpose

This repository contains a typed Python pipeline that:

1. Searches NewsAPI for recent articles about a company.
2. Restricts results to a maintained list of financial news domains.
3. Extracts and summarizes articles with `newspaper3k` and NLTK.
4. Classifies the title and summary with a financial sentiment model.
5. Returns structured article results and explicit extraction failures as JSON.

The current output is news sentiment, not a prediction of company or share-price
performance. Do not make predictive claims until the planned backtest exists.

## Repository Layout

```text
src/news_signal/
  entrypoints/
    cli.py         # argparse CLI and JSON output
    tools.py       # reusable agent-compatible callable
  adapters.py      # external service and model integrations
  application.py   # constructs the production pipeline
  config.py        # environment configuration
  models.py        # typed immutable result models
  pipeline.py      # protocols and application orchestration
tests/             # offline pytest tests
```

Other important files:

- `pyproject.toml` defines the package, dependency groups, pytest configuration,
  and `news-signal` console command.
- `uv.lock` records the resolved, reproducible dependency environment.
- `.python-version` pins project commands to Python 3.12.
- `.env.example` documents safe runtime configuration.
- `Sentences_50Agree.txt` contains 300 labelled Financial PhraseBank examples.
- `NewsSentimentAnalysis.py` is a compatibility wrapper for the CLI; new code
  should import `news_signal` or use the console command.
- `docs/PLAN.MD` records the sequential modernization plan.

## Runtime Flow

```text
CLI or tool arguments + environment
    -> NewsApiProvider.fetch
    -> NewspaperArticleExtractor.extract
    -> HuggingFaceSentimentClassifier.classify
    -> NewsAnalysisPipeline result
    -> dictionary for a tool, or JSON to stdout
```

`NewsAnalysisPipeline` depends on the `NewsProvider`, `ArticleExtractor`, and
`SentimentClassifier` protocols. Keep those boundaries injectable: production
uses the real adapters while tests use small fakes.

The pipeline deduplicates identical URLs, continues after known extraction
failures, and stops after the requested number of successful articles. Provider
and classifier failures currently propagate to the caller.

## Setup and Commands

Use `uv` for environment and dependency management:

```bash
uv sync
cp .env.example .env
uv run python -m nltk.downloader punkt punkt_tab
```

Run an analysis:

```bash
uv run news-signal analyse --company "NVIDIA" --limit 5 --days 7
```

Create a reusable agent tool:

```python
from news_signal.entrypoints.tools import build_tools

tools = build_tools()
result = tools.analyse_company_news("NVIDIA", limit=5)
```

Build the tool object once per agent process. It retains the pipeline and avoids
reconstructing the lazy-loaded classifier for each tool call.

Run the offline unit suite:

```bash
uv run pytest
```

Use `uv add`, `uv remove`, and `uv lock` for dependency changes, and `uv build`
for package artifacts. Do not introduce pip requirements files alongside the
`pyproject.toml` and `uv.lock` sources of truth.

The first production analysis can download the configured Hugging Face model.
It also makes requests to NewsAPI and publisher websites. Tests and imports must
remain offline and free of model or NLTK downloads.

## Configuration

`Settings.from_env()` loads `.env` from the current working directory and
validates:

- `NEWS_API_KEY`: required NewsAPI credential.
- `NEWS_LOOKBACK_DAYS`: optional positive integer, default `7`.
- `SENTIMENT_MODEL`: optional Hugging Face model identifier.

Never commit `.env`, `config.py`, credentials, or downloaded article bodies.
The old ignored `config.py` is no longer used by the application.

## Model and Evaluation Data

The default model is:

```text
mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
```

The classifier retains both the normalized label and model confidence, and
enables input truncation. Model loading is lazy.

`Sentences_50Agree.txt` uses `sentence@label` rows. The historical repository
and CV claim 92% accuracy, but that figure has not yet been reproduced against
the packaged implementation. Step 2 must record model revision, dataset version,
per-class metrics, confusion matrix, latency, and failures.

## Development Rules

- Keep imports fast and side-effect free.
- Keep external services behind the existing protocols.
- Use typed result models instead of pandas rows or unstructured dictionaries.
- Add focused pytest coverage for behavior and failure paths.
- Do not silently swallow provider or inference errors.
- Preserve source URLs, timestamps, confidence, and failure details.
- Do not globally disable TLS verification or download resources at import time.
- Keep deterministic scoring separate from model-generated output.
- Keep CLI, tool, and future API translation inside `entrypoints/`; do not put
  transport-specific behavior in the pipeline.

## Current Limitations

- Article processing is sequential and has no application-level retry or cache.
- Article extraction remains dependent on publisher markup, access restrictions,
  and local NLTK tokenizer data.
- The source allowlist is hard-coded.
- Only article-extraction failures are represented as partial results.
- There is no HTTP API, persistence, tracing, deployment, event extraction, company
  signal aggregation, or performance backtest yet.
- The Financial PhraseBank evaluator has not been migrated into the package.

Continue with one numbered step from `docs/PLAN.MD` at a time. Step 1 is
complete; Step 2 is the next planned change.
