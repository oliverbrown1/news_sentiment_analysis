from datetime import datetime, timezone

from company_signals.providers import (
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    SecCompanyResolver,
    SecFilingProvider,
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


def test_sec_company_resolver_matches_name_and_rejects_wrong_ticker() -> None:
    client = FakeClient(
        {
            SEC_TICKERS_URL: {
                "0": {"title": "NVIDIA CORP", "ticker": "NVDA", "cik_str": 1},
                "1": {"title": "TESLA INC", "ticker": "TSLA", "cik_str": 2},
            }
        }
    )
    resolver = SecCompanyResolver("Oliver test@example.com", client)

    assert resolver.find("NVIDIA", "nvda")[0].ticker == "NVDA"
    assert resolver.find("NVDA")[0].company == "NVIDIA CORP"
    assert resolver.find("NVIDIA", "TSLA") == []
