from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Protocol

from google.genai.errors import APIError

from eval.market_dataset import (
    MarketDataset,
    MarketDatasetExample,
    direction_for_return,
)
from eval.metrics import classification_metrics, environment_versions
from eval.models import ClassificationPrediction, ScoredPrediction
from market_signal_agent.evaluation import build_evaluation_runner, evaluate
from market_signal_agent.models import ForecastRequest, MarketForecast


class AgentSystem(Protocol):
    async def predict(self, example: MarketDatasetExample) -> MarketForecast: ...


@dataclass(frozen=True, slots=True)
class AgentEvaluationFailure:
    example_id: str
    ticker: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentEvaluationReport:
    schema_version: int
    generated_at: str
    task: dict[str, object]
    dataset: dict[str, object]
    model: dict[str, object]
    metrics: dict[str, object]
    review_samples: tuple[dict[str, object], ...]
    largest_errors: tuple[dict[str, object], ...]
    latency: dict[str, object]
    failures: tuple[AgentEvaluationFailure, ...]
    environment: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Scored:
    example: MarketDatasetExample
    forecast: MarketForecast


class AdkAgentSystem:
    def __init__(self, model: str) -> None:
        self.model = model
        self._runner = build_evaluation_runner(model)

    async def predict(self, example: MarketDatasetExample) -> MarketForecast:
        return await evaluate(
            self._runner,
            ForecastRequest(
                company=example.company,
                ticker=example.ticker,
                benchmark=example.benchmark,
                cutoff_date=example.published_at,
                headline=example.headline,
                headline_source=example.source,
                headline_url=example.url,
                forecast_horizon=example.forecast_horizon,
            ),
        )


async def evaluate_market_agent(
    system: AgentSystem,
    dataset: MarketDataset,
    *,
    system_name: str,
    model_name: str,
    limit: int | None = None,
) -> AgentEvaluationReport:
    if limit is not None and limit < 1:
        raise ValueError("agent evaluation limit must be at least 1")
    examples = _select_examples(dataset.examples, limit)
    scored: list[_Scored] = []
    failures: list[AgentEvaluationFailure] = []
    inference_times: list[float] = []
    # agent will predict sentiment and expected next day return expressed as percentage
    for example in examples:
        started = perf_counter()
        try:
            forecast = await system.predict(example)
        except (APIError, RuntimeError, ValueError) as exc:
            failures.append(
                AgentEvaluationFailure(example.example_id, example.ticker, str(exc))
            )
            continue
        inference_times.append(perf_counter() - started)
        scored.append(_Scored(example, forecast))

    if not scored:
        raise RuntimeError("agent evaluation produced no predictions")

    classification = classification_metrics(
        [
            ScoredPrediction(
                item.example.expected_direction,
                ClassificationPrediction(
                    direction_for_return(
                        item.forecast.predicted_return,
                        item.example.negative_threshold,
                        item.example.positive_threshold,
                    ),
                    1.0,
                ),
            )
            for item in scored
        ],
        count_key="evaluated_examples",
    )
    errors = [
        abs(item.forecast.predicted_return - item.example.target_return)
        for item in scored
    ]
    direction_matches = [
        _sign(item.forecast.predicted_return) == _sign(item.example.target_return)
        for item in scored
    ]
    return AgentEvaluationReport(
        schema_version=2,
        generated_at=datetime.now(timezone.utc).isoformat(),
        task={
            "name": "headline-level next-trading-day return forecasting",
            "input": "one real headline plus point-in-time market and filing tools",
            "target": "next trading session close versus the reference session close",
            "return_unit": "decimal",
            "class_thresholds": (
                "ticker-specific 30th and 60th percentiles of prior daily returns"
            ),
        },
        dataset={
            "name": "recent market headline dataset",
            "path": dataset.path,
            "sha256": dataset.sha256,
            "generated_at": dataset.generated_at,
            "seed_path": dataset.seed_path,
            "seed_sha256": dataset.seed_sha256,
            "examples": len(dataset.examples),
            "examples_selected": len(examples),
            "build_failures": len(dataset.build_failures),
            "date_start": min(
                item.published_at for item in dataset.examples
            ).isoformat(),
            "date_end": max(
                item.published_at for item in dataset.examples
            ).isoformat(),
        },
        model={"system": system_name, "name": model_name},
        metrics={
            "mean_absolute_error": mean(errors),
            "directional_accuracy": mean(direction_matches),
            "failure_rate": len(failures) / len(examples),
            "derived_classification": classification,
        },
        review_samples=_review_samples(examples, scored),
        largest_errors=_largest_errors(scored),
        latency={
            "examples_succeeded": len(inference_times),
            "mean_seconds_per_example": mean(inference_times),
        },
        failures=tuple(failures),
        environment=environment_versions(),
    )


def _review_samples(
    examples: tuple[MarketDatasetExample, ...], scored: list[_Scored]
) -> tuple[dict[str, object], ...]:
    scored_by_id = {item.example.example_id: item for item in scored}
    samples: list[dict[str, object]] = []
    for example in examples:
        if not example.review:
            continue
        item = scored_by_id.get(example.example_id)
        if item is None:
            samples.append(
                {
                    "example_id": example.example_id,
                    "ticker": example.ticker,
                    "headline": example.headline,
                    "output_valid": False,
                }
            )
            continue
        samples.append(
            {
                "example_id": example.example_id,
                "ticker": example.ticker,
                "headline": example.headline,
                "output_valid": True,
                "ticker_preserved": item.forecast.ticker == example.ticker,
                "citations_present": bool(item.forecast.evidence),
                "predicted_return": item.forecast.predicted_return,
                "target_return": example.target_return,
                "thesis": item.forecast.thesis,
                "risks": item.forecast.risks,
                "evidence": [entry.model_dump() for entry in item.forecast.evidence],
            }
        )
    return tuple(samples)


def _select_examples(
    examples: tuple[MarketDatasetExample, ...], limit: int | None
) -> tuple[MarketDatasetExample, ...]:
    if limit is None:
        return examples
    review = [example for example in examples if example.review]
    remaining = [example for example in examples if not example.review]
    return tuple((review + remaining)[:limit])


def _largest_errors(scored: list[_Scored]) -> tuple[dict[str, object], ...]:
    largest = sorted(
        scored,
        key=lambda item: abs(
            item.forecast.predicted_return - item.example.target_return
        ),
        reverse=True,
    )[:20]
    return tuple(
        {
            "example_id": item.example.example_id,
            "ticker": item.example.ticker,
            "published_at": item.example.published_at.isoformat(),
            "headline": item.example.headline,
            "predicted_return": item.forecast.predicted_return,
            "target_return": item.example.target_return,
            "absolute_error": abs(
                item.forecast.predicted_return - item.example.target_return
            ),
        }
        for item in largest
    )


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
