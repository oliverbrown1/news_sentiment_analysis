from datetime import datetime, timezone
from types import SimpleNamespace

from company_signals.providers import (
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    SecFilingProvider,
    YFinanceCompanyResolver,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
        self.headers = headers
        return FakeResponse(self.responses[url])


def test_sec_provider_selects_latest_filing_available_before_cutoff_date() -> None:
    submissions_url = f"{SEC_SUBMISSIONS_URL}/CIK0000000123.json"
    client = FakeClient(
        {
            SEC_TICKERS_URL: {"0": {"ticker": "EXM", "cik_str": 123}},
            submissions_url: {
                "filings": {
                    "recent": {
                        "form": ["10-Q", "8-K"],
                        "filingDate": ["2024-01-10", "2024-02-10"],
                        "acceptanceDateTime": [
                            "2024-01-10T16:00:00-05:00",
                            "2024-02-10T16:00:00-05:00",
                        ],
                        "accessionNumber": ["0001-24-000001", "0001-24-000002"],
                        "primaryDocument": ["q1.htm", "event.htm"],
                    },
                    "files": [],
                }
            },
        }
    )
    provider = SecFilingProvider("Oliver test@example.com", client)

    filing = provider.latest(
        "EXM", datetime(2024, 2, 1, tzinfo=timezone.utc), ("10-Q", "8-K")
    )

    assert filing is not None
    assert filing.form == "10-Q"
    assert filing.accession_number == "0001-24-000001"
    assert client.headers["User-Agent"] == "Oliver test@example.com"


def test_yfinance_company_resolver_accepts_exchange_qualified_ticker() -> None:
    calls: list[str] = []

    def searcher(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
        return SimpleNamespace(
            quotes=[
                {
                    "quoteType": "EQUITY",
                    "symbol": "IAG.L",
                    "longname": "International Consolidated Airlines Group, S.A.",
                }
            ]
        )

    resolver = YFinanceCompanyResolver(searcher)
    match = resolver.find("International Airlines Group", "iag.l")[0]

    assert calls == ["IAG.L"]
    assert match.company == "International Consolidated Airlines Group, S.A."
    assert match.ticker == "IAG.L"


def test_yfinance_company_resolver_rejects_mismatched_company_and_ticker() -> None:
    resolver = YFinanceCompanyResolver(
        lambda query, **kwargs: SimpleNamespace(
            quotes=[
                {
                    "quoteType": "EQUITY",
                    "symbol": "IAG",
                    "longname": "IAMGOLD Corporation",
                }
            ]
        )
    )

    assert resolver.find("International Airlines Group", "IAG") == []
