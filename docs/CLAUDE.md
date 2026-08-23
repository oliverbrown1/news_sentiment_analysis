# News Sentiment Analysis: Project Guide

Last reviewed: 2026-08-23

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
  application.py   # constructs production adapters and pipeline
  cli.py           # argparse CLI and JSON output
  config.py        # environment configuration
  extraction.py    # newspaper3k adapter
  interfaces.py    # typed Protocol contracts
  models.py        # typed immutable result models
  pipeline.py      # application orchestration
  providers.py     # NewsAPI adapter and source allowlist
  sentiment.py     # lazy Hugging Face classifier
tests/             # offline pytest tests
```

Other important files:

- `pyproject.toml` defines the package, dependencies, pytest configuration, and
  `news-signal` console command.
- `.env.example` documents safe runtime configuration.
- `Sentences_50Agree.txt` contains 300 labelled Financial PhraseBank examples.
- `NewsSentimentAnalysis.py` is a compatibility wrapper for the CLI; new code
  should import `news_signal` or use the console command.
- `docs/PLAN.MD` records the sequential modernization plan.

## Runtime Flow

```text
CLI arguments + environment
    -> NewsApiProvider.fetch
    -> NewspaperArticleExtractor.extract
    -> HuggingFaceSentimentClassifier.classify
    -> NewsAnalysisPipeline result
    -> JSON to stdout
```

`NewsAnalysisPipeline` depends on the `NewsProvider`, `ArticleExtractor`, and
`SentimentClassifier` protocols. Keep those boundaries injectable: production
uses the real adapters while tests use small fakes.

The pipeline deduplicates identical URLs, continues after known extraction
failures, and stops after the requested number of successful articles. Provider
and classifier failures currently propagate to the caller.

## Setup and Commands

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python -m nltk.downloader punkt punkt_tab
```

Run an analysis:

```bash
news-signal analyse --company "NVIDIA" --limit 5 --days 7
```

Run the offline unit suite:

```bash
pytest
```

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
- Do not add agent or API abstractions before the relevant plan step.

## Current Limitations

- Article processing is sequential and has no application-level retry or cache.
- Article extraction remains dependent on publisher markup, access restrictions,
  and local NLTK tokenizer data.
- The source allowlist is hard-coded.
- Only article-extraction failures are represented as partial results.
- There is no API, persistence, tracing, deployment, event extraction, company
  signal aggregation, or performance backtest yet.
- The Financial PhraseBank evaluator has not been migrated into the package.

Continue with one numbered step from `docs/PLAN.MD` at a time. Step 1 is
complete; Step 2 is the next planned change.
