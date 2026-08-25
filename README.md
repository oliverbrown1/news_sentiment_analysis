# Financial News Sentiment Analysis

A typed Python pipeline that retrieves recent financial news for a company,
extracts article summaries, and classifies their sentiment with a financial
DistilRoBERTa model: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`.

## Setup

Install `uv`, then sync the locked Python 3.12 environment:

```bash
uv sync
uv run python -m nltk.downloader punkt punkt_tab
cp .env.example .env
```

Obtain an API key from `https://newsapi.org/` and set `NEWS_API_KEY` in `.env`.
The optional settings control the default lookback period and Hugging Face model.

## Usage

```bash
uv run news-signal analyse --company "NVIDIA" --limit 5
uv run news-signal analyse --company "Rolls-Royce Holdings" --days 14

# evaluation also possible
uv run news-signal evaluate-sentiment --output reports/finentity-sentiment.json
```

The same pipeline can be exposed as a typed agent tool:

```python
from news_signal.entrypoints.tools import build_tools

tools = build_tools()
result = tools.analyse_company_news("NVIDIA", limit=5)
```

Create `tools` once when the agent process starts so repeated calls reuse the
same pipeline and loaded sentiment model.

The first real analysis may download the configured Hugging Face model, pending fix to this cold start issue. 
## Evaluation and Testing

The classifier can be evaluated (see Usage section), uses FinEntity dataset containing short financial summaries and labelled sentiment.

Reports dataset checksum, model revision, metrics (accuracy, F1, recall, per sentiment class metrics, confusion matrix, mean confidence, mean latency)

For structural/integrity unit tests, execute the below

```bash
uv run pytest
```

The unit tests use fake providers and classifiers, so they do not need an API key, network access, NLTK data, or model download.

## How It Works

1. Scrape news articles by default in the last 7 days using newsapi (requires API key, see Setup) and matches against pre-defined Financial Domains list
2. Perform extraction of individual articles using newspaper3k and summarise body content using NLTK
3. Perform sentiment analysis on summary using DistilRoBERTa model from Hugging Face -> "positive", "neutral", "negative" result

## Backlog

* 300-row Financial PhraseBank subset retained as a legacy regression benchmark within `Sentences_50Agree.txt`
* Review Python libraries used for scraping (newsapi), extraction (newspaper3k), summarisation (NLTK) and sentiment analysis (DistilRoBERTa)
* Use sentiment output and other signals to predict company performance, using Agentic AI pipeline
* Use FinMarBa for later evaluation of the agentic pipeline - evaluation dataset which shows how market performs/move after headlines, not just sentiment.
