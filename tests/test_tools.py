from news_signal_v1.entrypoints.tools import NewsSignalTools
from news_signal_v1.models import AnalysisResult


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def analyse(self, company: str, limit: int, lookback_days: int) -> AnalysisResult:
        self.calls.append((company, limit, lookback_days))
        return AnalysisResult(company=company, articles=())


def test_tool_reuses_pipeline_and_returns_serializable_result() -> None:
    pipeline = FakePipeline()
    tools = NewsSignalTools(pipeline, default_lookback_days=14)

    first = tools.analyse_company_news("Example Ltd")
    second = tools.analyse_company_news("Another Ltd", limit=2, lookback_days=30)

    assert first["company"] == "Example Ltd"
    assert second["company"] == "Another Ltd"
    assert pipeline.calls == [
        ("Example Ltd", 5, 14),
        ("Another Ltd", 2, 30),
    ]
