from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

SignalGroup = Literal[
    "news",
    "price_momentum",
    "market_activity",
    "relative_performance",
    "fundamental",
]
SignalValue = float | int | str
EvidenceKind = Literal["news", "filing"]


class SignalProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompanyMatch:
    company: str
    ticker: str


@dataclass(frozen=True, slots=True)
class PriceBar:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def available_at(self) -> datetime:
        return datetime.combine(
            self.session_date + timedelta(days=1), time.min, timezone.utc
        )


@dataclass(frozen=True, slots=True)
class Filing:
    form: str
    accepted_at: datetime
    accession_number: str
    url: str


@dataclass(frozen=True, slots=True)
class Signal:
    group: SignalGroup
    name: str
    value: SignalValue
    observed_at: datetime
    available_at: datetime
    source: str
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.available_at.tzinfo is None:
            raise ValueError("signal timestamps must include a timezone")
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot be before observed_at")


@dataclass(frozen=True, slots=True)
class SignalFailure:
    source: str
    stage: str
    reason: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    kind: EvidenceKind
    title: str
    url: str
    available_at: datetime
    excerpt: str | None = None
    sentiment: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.available_at.tzinfo is None:
            raise ValueError("evidence timestamp must include a timezone")


@dataclass(frozen=True, slots=True)
class NewsStats:
    eligible: int
    attempted: int
    analysed: int
    limit: int


@dataclass(frozen=True, slots=True)
class SignalResult:
    signals: tuple[Signal, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    failures: tuple[SignalFailure, ...] = ()
    # only for news
    news_stats: NewsStats | None = None

    def to_dict(
        self,
        *,
        include_excerpts: bool = True,
        verbose_failures: bool = False,
    ) -> dict[str, object]:
        return _serialise_result(
            asdict(self),
            include_excerpts=include_excerpts,
            verbose_failures=verbose_failures,
        )


@dataclass(frozen=True, slots=True)
class SignalBundle:
    company: str
    ticker: str
    cutoff_date: datetime
    benchmark: str
    signals: tuple[Signal, ...]
    news_stats: NewsStats
    evidence: tuple[EvidenceReference, ...] = ()
    failures: tuple[SignalFailure, ...] = ()

    def to_dict(self, *, verbose: bool = False) -> dict[str, object]:
        return _serialise_result(
            asdict(self),
            include_excerpts=verbose,
            verbose_failures=verbose,
        )


def _serialise_result(
    result: dict[str, object],
    *,
    include_excerpts: bool,
    verbose_failures: bool,
) -> dict[str, object]:
    if result.get("news_stats") is None:
        result.pop("news_stats", None)

    evidence = result["evidence"]
    if isinstance(evidence, (list, tuple)):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if not include_excerpts:
                item.pop("excerpt", None)
            for key in tuple(item):
                if item[key] is None:
                    item.pop(key)

    if not verbose_failures:
        failures = result["failures"]
        summaries: dict[tuple[str, str], int] = {}
        if isinstance(failures, (list, tuple)):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                source = failure.get("source")
                stage = failure.get("stage")
                if not isinstance(source, str) or not isinstance(stage, str):
                    continue
                if source == "news" and stage != "fetch":
                    continue
                key = (source, stage)
                summaries[key] = summaries.get(key, 0) + 1
        result["failures"] = [
            {"source": source, "stage": stage, "count": count}
            for (source, stage), count in summaries.items()
        ]
    return result
