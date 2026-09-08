# Agentic Market Forecast

Google ADK agent that "forecasts" next-day return using grounded market data and news signals, leveraging Hugging Face models for sentiment analysis and yfinance for market data.


## Setup

Install `uv`, then sync the locked Python 3.12 environment:

```bash
uv sync
uv run python -m nltk.downloader punkt punkt_tab
cp .env.example .env
```

To populate the .env:
1. Obtain an API key from `https://newsapi.org/` and set `NEWS_API_KEY` in `.env`.
2. `GOOGLE_API_KEY` and `SEC_USER_AGENT` (your identity) are rqeuired fields to setup the agent.

## Usage

```bash
# V2 is the default command
uv run news-signal analyse --company "NVIDIA" --ticker NVDA

# Explicit, comparable entry points
uv run news-signal-v1 analyse --company "Rolls-Royce Holdings"
uv run news-signal-v2 analyse --company "Rolls-Royce Holdings"
```

## Agentic Workflow

Agent does not do collect data on its own through tools like Web Search.

Instead specialised tools are given to fetch real market data and signals (prince momentum, volume, benchmark performance, SEC metadata etc.) as well as news sentiment and signals (see below). 

The agent will be responsible for calling these tools with the right arguments, search terms, company verification and interpeting the data given back to it. It then uses this data to provide an accurate next-day return prediction.

## News Analysis

2 versions of the pipeline which scrapes relevant news articles on a company, analyses sentiment.

### V1 vs V2

- **Ingestion:** V1 uses the `newsapi-python` SDK with a hard-coded domain allowlist; V2 uses direct `httpx` requests with optional ticker and domain filtering.
- **Article processing:** V1 uses `newspaper3k` and NLTK to create a general summary; V2 uses Trafilatura to extract the article body, then selects sentences mentioning the target company or ticker.
- **Sentiment:** V1 uses the legacy DistilRoBERTa model; V2 uses ModernFinBERT and improving accuracy in evaluations from 69.03% to 75.41% and macro F1 from 69.58% to 75.38%.


## Evaluation

The model-independent `eval` package contains 3 task-specific evaluators:

- `sentiment_eval` uses FinEntity to evaluate semantic financial sentiment.
- `market_eval` uses FinMarBa to evaluate market-direction predictions (next-day return) for dated headlines, sentiment based on market direction.
- `agent_eval` uses a custom dataset instead of `market_eval` since FinMarBa data is from 2010-2011 whilst NewsAPI goes back 5 years 

Example commands:

```bash
uv run news-signal-eval --task sentiment --system v2 # sentiment analysis eval
uv run news-signal-eval --task market --system v2 # agent eval with finmarba
uv run news-signal-eval --task agent --system agent # agent eval with custom dataset
```

Evaluators report the dataset checksum, model revision, accuracy, macro F1,
per-class metrics, confusion matrix, calibration, latency, and high-confidence
errors. 

See `data/README.md` for more info on evaluation data and how its created/used.
 

## Testing

For structural/integrity unit tests, execute the below

```bash
uv run pytest
```

Unit tests use fakes and need no credentials, network, NLTK data, or model download.

## Backlog

- Improve the agent from evaluation results
- Add a genai-based `news_signal_v3` and compare it with V1 and V2 on the same sentiment benchmark.
- Expose FastAPI interfaces with tracing, CI and reproducible evaluation reports.