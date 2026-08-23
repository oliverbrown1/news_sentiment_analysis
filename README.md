# Financial News Sentiment Analysis

A typed Python pipeline that retrieves recent financial news for a company,
extracts article summaries, and classifies their sentiment with a financial
DistilRoBERTa model.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python -m nltk.downloader punkt punkt_tab
```

Set `NEWS_API_KEY` in `.env`. The optional settings in `.env.example` control
the default lookback period and Hugging Face model.

## Usage

```bash
news-signal analyse --company "NVIDIA" --limit 5
news-signal analyse --company "Rolls-Royce Holdings" --days 14
```

The command prints structured JSON containing the article metadata, extracted
summary, sentiment label and confidence. Extraction failures are included
separately so a partially successful run remains useful.

The first real analysis may download the configured Hugging Face model. Article
extraction and NewsAPI access also require network connectivity.

## Tests

```bash
pytest
```

The unit tests use fake providers and classifiers, so they do not need an API
key, network access, NLTK data, or model download.

## Structure

```text
src/news_signal/
  application.py   # production dependency wiring
  cli.py           # terminal interface
  config.py        # validated environment settings
  extraction.py    # newspaper3k article extraction
  interfaces.py    # pipeline dependency contracts
  models.py        # typed inputs and results
  pipeline.py      # fetch, extract and classify orchestration
  providers.py     # NewsAPI adapter
  sentiment.py     # Hugging Face classifier adapter
tests/             # offline pytest unit tests
```

The classifier uses
`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`. The
repository also contains a 300-row Financial PhraseBank subset for the
reproducible evaluation work planned next.
