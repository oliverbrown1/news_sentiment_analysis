from news_signal_v2.entrypoints.tools import NewsSignalTools
from news_signal_v2.models import AnalysisResult


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def analyse(self, **kwargs) -> AnalysisResult:
        self.calls.append(kwargs)
        return AnalysisResult(
            company=str(kwargs["company"]),
            ticker=kwargs["ticker"],
            articles=(),
        )


def test_v2_tool_reuses_pipeline_and_returns_serializable_result() -> None:
    pipeline = FakePipeline()
    tools = NewsSignalTools(pipeline, default_lookback_days=14)

    result = tools.analyse_company_news("Example Ltd", ticker="EXM", limit=2)

    assert result["company"] == "Example Ltd"
    assert pipeline.calls == [
        {
            "company": "Example Ltd",
            "ticker": "EXM",
            "limit": 2,
            "lookback_days": 14,
        }
    ]
