from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from company_signals.models import PriceBar
from company_signals.providers import PriceProvider
from eval.models import ClassificationLabel, EvaluationDataError
from news_signal_v2.models import NewsSearchResult

SEED_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 2
DEFAULT_FORECAST_HORIZON = "next trading day"


class NewsProvider(Protocol):
    def fetch(
        self,
        search_terms: tuple[str, ...],
        ticker: str | None,
        lookback_days: int,
        cutoff_date: datetime | None = None,
    ) -> NewsSearchResult: ...


@dataclass(frozen=True, slots=True)
class MarketSeed:
    seed_id: str
    company: str
    ticker: str
    benchmark: str
    cutoff_date: datetime
    news_terms: tuple[str, ...]
    lookback_days: int = 7
    article_limit: int = 20
    review: bool = False


@dataclass(frozen=True, slots=True)
class MarketSeedFile:
    path: str
    sha256: str
    seeds: tuple[MarketSeed, ...]


@dataclass(frozen=True, slots=True)
class MarketDatasetExample:
    example_id: str
    seed_id: str
    company: str
    ticker: str
    benchmark: str
    published_at: datetime
    headline: str
    source: str
    url: str
    forecast_horizon: str
    target_return: float
    negative_threshold: float
    positive_threshold: float
    expected_direction: ClassificationLabel
    reference_session: str
    target_session: str
    review: bool = False

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["record_type"] = "example"
        record["published_at"] = self.published_at.isoformat()
        return record


@dataclass(frozen=True, slots=True)
class MarketDatasetFailure:
    seed_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class MarketDataset:
    path: str
    sha256: str
    generated_at: str
    seed_path: str
    seed_sha256: str
    examples: tuple[MarketDatasetExample, ...]
    build_failures: tuple[MarketDatasetFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class _Target:
    value: float
    negative_threshold: float
    positive_threshold: float
    direction: ClassificationLabel
    reference_session: str
    target_session: str


def load_market_seed(path: Path) -> MarketSeedFile:
    raw = _read_bytes(path, "market evaluation seed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"invalid market evaluation seed: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SEED_SCHEMA_VERSION
    ):
        raise EvaluationDataError("market evaluation seed has an unsupported schema")
    rows = payload.get("seeds")
    if not isinstance(rows, list) or not rows:
        raise EvaluationDataError("market evaluation seed contains no entries")

    seeds: list[MarketSeed] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        try:
            seed = _parse_seed(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationDataError(f"invalid market seed {index}: {exc}") from exc
        if seed.seed_id in seen_ids:
            raise EvaluationDataError(f"duplicate market seed id: {seed.seed_id}")
        seen_ids.add(seed.seed_id)
        seeds.append(seed)
    return MarketSeedFile(str(path), hashlib.sha256(raw).hexdigest(), tuple(seeds))


def build_market_dataset(
    seed_file: MarketSeedFile,
    news_provider: NewsProvider,
    price_provider: PriceProvider,
) -> MarketDataset:
    examples: list[MarketDatasetExample] = []
    failures: list[MarketDatasetFailure] = []
    seen_articles: set[tuple[str, str]] = set()

    for seed in seed_file.seeds:
        try:
            search = news_provider.fetch(
                seed.news_terms,
                seed.ticker,
                seed.lookback_days,
                seed.cutoff_date,
            )
            articles = [
                article
                for article in search.articles
                if article.published_at is not None
                and _normalise_datetime(article.published_at) <= seed.cutoff_date
            ]
            if not articles:
                raise EvaluationDataError("no dated headlines were returned")

            first_date = min(
                cast(datetime, article.published_at).date() for article in articles
            )
            bars = _price_history(seed, first_date, price_provider)
        except (EvaluationDataError, RuntimeError, ValueError) as exc:
            failures.append(MarketDatasetFailure(seed.seed_id, str(exc)))
            continue

        review_assigned = False
        accepted = 0
        for article in articles:
            if accepted >= seed.article_limit:
                break
            published_at = _normalise_datetime(cast(datetime, article.published_at))
            url = _canonical_url(article.url)
            duplicate_key = (seed.ticker, url)
            if duplicate_key in seen_articles:
                continue
            try:
                target = _target_for(seed.ticker, published_at.date(), bars)
            except EvaluationDataError as exc:
                failures.append(MarketDatasetFailure(seed.seed_id, str(exc)))
                continue

            seen_articles.add(duplicate_key)
            accepted += 1
            review = seed.review and not review_assigned
            review_assigned = review_assigned or review
            examples.append(
                MarketDatasetExample(
                    example_id=_example_id(seed.ticker, published_at, url),
                    seed_id=seed.seed_id,
                    company=seed.company,
                    ticker=seed.ticker,
                    benchmark=seed.benchmark,
                    published_at=published_at,
                    headline=article.title.strip(),
                    source=article.source_name.strip(),
                    url=url,
                    forecast_horizon=DEFAULT_FORECAST_HORIZON,
                    target_return=target.value,
                    negative_threshold=target.negative_threshold,
                    positive_threshold=target.positive_threshold,
                    expected_direction=target.direction,
                    reference_session=target.reference_session,
                    target_session=target.target_session,
                    review=review,
                )
            )
        if accepted == 0:
            failures.append(
                MarketDatasetFailure(seed.seed_id, "no usable unique headlines")
            )

    if not examples:
        details = "; ".join(
            f"{failure.seed_id}: {failure.reason}" for failure in failures
        )
        raise EvaluationDataError(
            f"market dataset build produced no examples. Failures: {details}"
        )
    return MarketDataset(
        path="",
        sha256="",
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed_path=seed_file.path,
        seed_sha256=seed_file.sha256,
        examples=tuple(examples),
        build_failures=tuple(failures),
    )


def write_market_dataset(dataset: MarketDataset, path: Path) -> None:
    metadata = {
        "record_type": "metadata",
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": dataset.generated_at,
        "seed_path": dataset.seed_path,
        "seed_sha256": dataset.seed_sha256,
        "build_failures": [asdict(failure) for failure in dataset.build_failures],
    }
    lines = [json.dumps(metadata, sort_keys=True)]
    lines.extend(
        json.dumps(example.to_record(), sort_keys=True)
        for example in dataset.examples
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_market_dataset(path: Path) -> MarketDataset:
    raw = _read_bytes(path, "market evaluation dataset")
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"invalid market evaluation dataset: {path}") from exc
    if not records or not isinstance(records[0], dict):
        raise EvaluationDataError("market evaluation dataset is empty")
    metadata = records[0]
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("schema_version") != DATASET_SCHEMA_VERSION
    ):
        raise EvaluationDataError("market evaluation dataset has an unsupported schema")

    examples: list[MarketDatasetExample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records[1:], start=2):
        try:
            example = _parse_example(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationDataError(
                f"invalid market dataset line {index}: {exc}"
            ) from exc
        if example.example_id in seen_ids:
            raise EvaluationDataError(
                f"duplicate market dataset example id: {example.example_id}"
            )
        seen_ids.add(example.example_id)
        examples.append(example)
    if not examples:
        raise EvaluationDataError("market evaluation dataset contains no examples")

    raw_failures = metadata.get("build_failures", [])
    if not isinstance(raw_failures, list):
        raise EvaluationDataError("market dataset build_failures must be a list")
    failures: list[MarketDatasetFailure] = []
    for item in raw_failures:
        if not isinstance(item, dict):
            raise EvaluationDataError("market dataset contains an invalid build failure")
        failures.append(
            MarketDatasetFailure(
                seed_id=_text(item, "seed_id"),
                reason=_text(item, "reason"),
            )
        )
    return MarketDataset(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        generated_at=_text(metadata, "generated_at"),
        seed_path=_text(metadata, "seed_path"),
        seed_sha256=_text(metadata, "seed_sha256"),
        examples=tuple(examples),
        build_failures=tuple(failures),
    )


def direction_for_return(
    value: float, negative_threshold: float, positive_threshold: float
) -> ClassificationLabel:
    if value < negative_threshold:
        return "negative"
    if value > positive_threshold:
        return "positive"
    return "neutral"


def _price_history(
    seed: MarketSeed, first_article_date: date, provider: PriceProvider
) -> list[PriceBar]:
    start = _years_before(first_article_date, 5)
    end = seed.cutoff_date.date() + timedelta(days=10)
    return sorted(provider.fetch(seed.ticker, start, end), key=lambda bar: bar.session_date)


def _target_for(ticker: str, published_date: date, bars: list[PriceBar]) -> _Target:
    reference = [bar for bar in bars if bar.session_date <= published_date]
    future = [bar for bar in bars if bar.session_date > published_date]
    if not reference:
        raise EvaluationDataError(
            f"no reference price available for {ticker} on {published_date}"
        )
    if not future:
        raise EvaluationDataError(
            f"no target price available for {ticker} after {published_date}"
        )

    historical_returns = _returns(
        [
            bar
            for bar in reference
            if bar.session_date >= _years_before(published_date, 5)
        ]
    )
    if len(historical_returns) < 252:
        raise EvaluationDataError(
            f"fewer than 252 historical returns are available for {ticker}"
        )
    negative_threshold = _percentile(historical_returns, 0.3)
    positive_threshold = _percentile(historical_returns, 0.6)
    target_return = future[0].close / reference[-1].close - 1
    if not math.isfinite(target_return):
        raise EvaluationDataError(f"invalid target return for {ticker}")
    return _Target(
        value=target_return,
        negative_threshold=negative_threshold,
        positive_threshold=positive_threshold,
        direction=direction_for_return(
            target_return, negative_threshold, positive_threshold
        ),
        reference_session=reference[-1].session_date.isoformat(),
        target_session=future[0].session_date.isoformat(),
    )


def _returns(bars: list[PriceBar]) -> list[float]:
    return [
        current.close / previous.close - 1
        for previous, current in zip(bars, bars[1:], strict=False)
        if previous.close > 0
    ]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _parse_seed(raw: object) -> MarketSeed:
    if not isinstance(raw, dict):
        raise TypeError("entry must be an object")
    terms = _text_list(raw, "news_terms")
    if not terms:
        raise ValueError("news_terms cannot be empty")
    if len(terms) > 5:
        raise ValueError("news_terms cannot contain more than five values")
    if any(not 2 <= len(term) <= 80 or '"' in term for term in terms):
        raise ValueError("news_terms contains an invalid value")
    return MarketSeed(
        seed_id=_text(raw, "seed_id"),
        company=_text(raw, "company"),
        ticker=_text(raw, "ticker").upper(),
        benchmark=_text(raw, "benchmark").upper(),
        cutoff_date=_datetime(_text(raw, "cutoff_date"), "cutoff_date"),
        news_terms=tuple(terms),
        lookback_days=_positive_int(
            raw.get("lookback_days", raw.get("news_days", 7)), "lookback_days"
        ),
        article_limit=_positive_int(
            raw.get("article_limit", raw.get("news_limit", 20)), "article_limit"
        ),
        review=_bool(raw.get("review", False), "review"),
    )


def _parse_example(raw: object) -> MarketDatasetExample:
    if not isinstance(raw, dict) or raw.get("record_type") != "example":
        raise ValueError("record_type must be example")
    target_return = _finite_float(raw.get("target_return"), "target_return")
    negative = _finite_float(raw.get("negative_threshold"), "negative_threshold")
    positive = _finite_float(raw.get("positive_threshold"), "positive_threshold")
    if negative >= positive:
        raise ValueError("return thresholds are not ordered")
    expected = direction_for_return(target_return, negative, positive)
    if raw.get("expected_direction") != expected:
        raise ValueError("expected_direction does not match target_return")
    return MarketDatasetExample(
        example_id=_text(raw, "example_id"),
        seed_id=_text(raw, "seed_id"),
        company=_text(raw, "company"),
        ticker=_text(raw, "ticker").upper(),
        benchmark=_text(raw, "benchmark").upper(),
        published_at=_datetime(_text(raw, "published_at"), "published_at"),
        headline=_text(raw, "headline"),
        source=_text(raw, "source"),
        url=_text(raw, "url"),
        forecast_horizon=_text(raw, "forecast_horizon"),
        target_return=target_return,
        negative_threshold=negative,
        positive_threshold=positive,
        expected_direction=expected,
        reference_session=_text(raw, "reference_session"),
        target_session=_text(raw, "target_session"),
        review=_bool(raw.get("review", False), "review"),
    )


def _example_id(ticker: str, published_at: datetime, url: str) -> str:
    value = f"{ticker}|{published_at.isoformat()}|{url}".encode()
    return hashlib.sha256(value).hexdigest()[:16]


def _canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvaluationDataError(f"could not read {label}: {path}") from exc


def _datetime(value: str, key: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{key} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _normalise_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise EvaluationDataError("headline publication time must include a timezone")
    return value.astimezone(timezone.utc)


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value.strip()


def _text_list(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return [cast(str, item).strip() for item in value]


def _positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _finite_float(value: object, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result
