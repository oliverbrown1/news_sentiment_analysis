# Financial News Signal

A typed Python pipeline that retrieves recent financial news for a company 
extracts article summaries, and classifies their sentiment. Currently 2 versions, using pre-trained open-source Sentiment Analysis models on Hugging Face.

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
# V2 is the default command
uv run news-signal analyse --company "NVIDIA" --ticker NVDA

# Explicit, comparable entry points
uv run news-signal-v1 analyse --company "Rolls-Royce Holdings"
uv run news-signal-v2 analyse --company "Rolls-Royce Holdings"
```

## V1 vs V2

- **Ingestion:** V1 uses the `newsapi-python` SDK with a hard-coded domain allowlist; V2 uses direct `httpx` requests with optional ticker and domain filtering.
- **Article processing:** V1 uses `newspaper3k` and NLTK to create a general summary; V2 uses Trafilatura to extract the article body, then selects sentences mentioning the target company or ticker.
- **Sentiment:** V1 uses the legacy DistilRoBERTa model; V2 uses ModernFinBERT and improves FinEntity accuracy from 69.03% to 75.41% and macro F1 from 69.58% to 75.38%.

## Evaluation

The model-independent `eval` package contains two task-specific evaluators:

- `sentiment_eval` uses FinEntity to evaluate semantic financial sentiment.
- `market_eval` uses FinMarBa to evaluate market-direction predictions for dated headlines and tickers, sentiment is dervied from market-direction.

V1 and V2 sentiment reports are generated with:

```bash
uv run news-signal-eval --system v1 --output reports/finentity-sentiment-v1.json
uv run news-signal-eval --system v2 --output reports/finentity-sentiment-v2.json
uv run news-signal-eval --task market --system v2 --output reports/finmarba-market-sentiment-v2.json
```

Both evaluators report the dataset checksum, model revision, accuracy, macro F1,
per-class metrics, confusion matrix, calibration, latency, and high-confidence
errors. 

The `market_eval` system will be used to evaluate the agent, but we can also evaluate against just the sentiment models as a baseline, since `market_eval` dataset also contains sentiment. See `data/market-sentiment-v2.json`

## Agent Tool

```python
from news_signal_v2.entrypoints.tools import build_tools

tools = build_tools()
result = tools.analyse_company_news(
    "NVIDIA", limit=5
)
```

Create `tools` once when the agent process starts so repeated calls reuse the
same pipeline and loaded sentiment model.

The first real analysis may download the configured Hugging Face model, pending fix to this cold start issue. 

## Testing

For structural/integrity unit tests, execute the below

```bash
uv run pytest
```

Unit tests use fakes and need no credentials, network, NLTK data, or model download.

## Backlog

* Scrape company signals for given date to enrich sentiment label for news articles and more accurately determine performance
* Use sentiment output and other signals to predict company performance, using Agentic AI pipeline
* Evaluate agentic pipeline against `market_eval` evaluation module.