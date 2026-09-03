from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from company_signals.models import Filing, PriceBar, SignalProviderError

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"


class PriceProvider(Protocol):
    @property
    def source(self) -> str: ...

    def fetch(self, ticker: str, start: date, end: date) -> list[PriceBar]: ...


class FilingProvider(Protocol):
    def latest(
        self, ticker: str, cutoff_date: datetime, forms: tuple[str, ...]
    ) -> Filing | None: ...


class YFinancePriceProvider:
    def __init__(self, downloader: Callable[..., Any] | None = None) -> None:
        if downloader is None:
            import yfinance

            downloader = yfinance.download
        self._download = downloader

    @property
    def source(self) -> str:
        return "yfinance"

    def fetch(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        try:
            frame = self._download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                progress=False,
                multi_level_index=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SignalProviderError(f"could not download prices for {ticker}") from exc
        if frame is None or frame.empty:
            raise SignalProviderError(f"no prices found for {ticker}")

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(frame.columns):
            raise SignalProviderError(f"price response for {ticker} is missing OHLCV data")

        bars: list[PriceBar] = []
        for index, row in frame.iterrows():
            session_date = (
                index.date()
                if hasattr(index, "date")
                else date.fromisoformat(str(index))
            )
            try:
                bars.append(
                    PriceBar(
                        session_date=session_date,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise SignalProviderError(
                    f"price response for {ticker} contains invalid OHLCV data"
                ) from exc
        return bars


class SecFilingProvider:
    def __init__(self, user_agent: str, client: Any | None = None) -> None:
        if not user_agent.strip():
            raise ValueError("SEC user agent cannot be empty")
        self._client = client or httpx.Client(timeout=20.0)
        self._headers = {"User-Agent": user_agent.strip(), "Accept": "application/json"}
        self._ciks: dict[str, int] | None = None

    def latest(
        self, ticker: str, cutoff_date: datetime, forms: tuple[str, ...]
    ) -> Filing | None:
        if cutoff_date.tzinfo is None:
            raise ValueError("cutoff_date must include a timezone")
        cik = self._cik_for(ticker)
        company = self._get_json(f"{SEC_SUBMISSIONS_URL}/CIK{cik:010d}.json")
        filings = company.get("filings")
        if not isinstance(filings, dict):
            raise SignalProviderError("SEC submissions response is missing filings")

        datasets: list[dict[str, Any]] = []
        recent = filings.get("recent")
        if isinstance(recent, dict):
            datasets.append(recent)
        files = filings.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict) or not _file_can_contain(
                    item, cutoff_date.date()
                ):
                    continue
                name = item.get("name")
                if isinstance(name, str) and name:
                    datasets.append(self._get_json(f"{SEC_SUBMISSIONS_URL}/{name}"))

        candidates = [
            filing
            for dataset in datasets
            for filing in _filings_from(dataset, cik)
            if filing.form in forms and filing.accepted_at <= cutoff_date
        ]
        return max(candidates, key=lambda filing: filing.accepted_at, default=None)

    def _cik_for(self, ticker: str) -> int:
        if self._ciks is None:
            payload = self._get_json(SEC_TICKERS_URL)
            self._ciks = {
                str(item["ticker"]).upper(): int(item["cik_str"])
                for item in payload.values()
                if isinstance(item, dict) and "ticker" in item and "cik_str" in item
            }
        try:
            return self._ciks[ticker.upper()]
        except KeyError as exc:
            raise SignalProviderError(f"SEC has no company mapping for {ticker}") from exc

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            response = self._client.get(url, headers=self._headers)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SignalProviderError(f"SEC request failed: {url}") from exc
        if not isinstance(payload, dict):
            raise SignalProviderError(f"SEC returned invalid data: {url}")
        return payload


def _file_can_contain(item: dict[str, Any], target: date) -> bool:
    start = item.get("filingFrom")
    end = item.get("filingTo")
    try:
        return date.fromisoformat(str(start)) <= target and date.fromisoformat(str(end)) >= target
    except ValueError:
        return True


def _filings_from(payload: dict[str, Any], cik: int) -> list[Filing]:
    forms = payload.get("form")
    dates = payload.get("filingDate")
    accepted = payload.get("acceptanceDateTime")
    accessions = payload.get("accessionNumber")
    documents = payload.get("primaryDocument")
    if not all(isinstance(values, list) for values in (forms, dates, accessions, documents)):
        return []

    filings: list[Filing] = []
    for index, (form, filing_date, accession, document) in enumerate(
        zip(forms, dates, accessions, documents, strict=False)
    ):
        available_at = _accepted_at(accepted, index, filing_date)
        accession_number = str(accession)
        accession_path = accession_number.replace("-", "")
        filings.append(
            Filing(
                form=str(form).upper(),
                accepted_at=available_at,
                accession_number=accession_number,
                url=f"{SEC_ARCHIVES_URL}/{cik}/{accession_path}/{document}",
            )
        )
    return filings


def _accepted_at(accepted: Any, index: int, filing_date: Any) -> datetime:
    if isinstance(accepted, list) and index < len(accepted) and accepted[index]:
        value = str(accepted[index]).replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(value)
            if result.tzinfo is None:
                result = result.replace(tzinfo=ZoneInfo("America/New_York"))
            return result.astimezone(timezone.utc)
        except ValueError:
            pass
    day = date.fromisoformat(str(filing_date))
    return datetime.combine(day + timedelta(days=1), time.min, timezone.utc)
