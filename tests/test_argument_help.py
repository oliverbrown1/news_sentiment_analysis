import inspect

import pytest

from company_signals.entrypoints.arguments import ARGUMENTS as COMPANY_ARGUMENTS
from company_signals.entrypoints.cli import build_parser as build_company_parser
from company_signals.entrypoints.tools import CompanySignalTools
from eval.arguments import ARGUMENTS as EVAL_ARGUMENTS
from eval.cli import build_parser as build_eval_parser
from market_signal_agent.entrypoints.cli import build_parser as build_agent_parser
from news_signal_v1.entrypoints.arguments import ARGUMENTS as V1_ARGUMENTS
from news_signal_v1.entrypoints.cli import build_parser as build_v1_parser
from news_signal_v1.entrypoints.tools import NewsSignalTools as V1Tools
from news_signal_v2.entrypoints.arguments import ARGUMENTS as V2_ARGUMENTS
from news_signal_v2.entrypoints.cli import build_parser as build_v2_parser
from news_signal_v2.entrypoints.tools import NewsSignalTools as V2Tools


@pytest.mark.parametrize(
    "parser_factory,command,descriptions",
    [
        (build_v1_parser, ["analyse", "--help"], V1_ARGUMENTS),
        (
            build_v2_parser,
            ["analyse", "--help"],
            {key: value for key, value in V2_ARGUMENTS.items() if key != "cutoff_date"},
        ),
        (build_company_parser, ["collect", "--help"], COMPANY_ARGUMENTS),
        (build_agent_parser, ["chat", "--help"], {}),
        (build_eval_parser, ["--help"], EVAL_ARGUMENTS),
    ],
)
def test_cli_help_uses_shared_argument_descriptions(
    parser_factory, command: list[str], descriptions: dict[str, str], capsys
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parser_factory().parse_args(command)

    assert exit_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert all(description in output for description in descriptions.values())


@pytest.mark.parametrize(
    "method,descriptions",
    [
        (V1Tools.analyse_company_news, V1_ARGUMENTS),
        (V2Tools.analyse_company_news, V2_ARGUMENTS),
        (CompanySignalTools.collect_signals, COMPANY_ARGUMENTS),
        (
            CompanySignalTools.find_companies,
            {key: COMPANY_ARGUMENTS[key] for key in ("company", "ticker")},
        ),
        (
            CompanySignalTools.get_news_signals,
            {
                key: COMPANY_ARGUMENTS[key]
                for key in ("company", "ticker", "cutoff_date", "news_days", "news_limit")
            },
        ),
        (
            CompanySignalTools.get_market_signals,
            {
                key: COMPANY_ARGUMENTS[key]
                for key in ("ticker", "cutoff_date", "benchmark", "price_days")
            },
        ),
        (
            CompanySignalTools.get_filing_metadata,
            {
                key: COMPANY_ARGUMENTS[key]
                for key in ("ticker", "cutoff_date")
            },
        ),
    ],
)
def test_tool_docstrings_use_shared_argument_descriptions(
    method, descriptions: dict[str, str]
) -> None:
    docstring = inspect.getdoc(method)

    assert docstring is not None
    assert all(description in docstring for description in descriptions.values())


def test_argument_descriptions_are_single_short_sentences() -> None:
    descriptions = {
        *V1_ARGUMENTS.values(),
        *V2_ARGUMENTS.values(),
        *COMPANY_ARGUMENTS.values(),
        *EVAL_ARGUMENTS.values(),
    }

    assert all(description.endswith(".") for description in descriptions)
    assert all(description.count(".") == 1 for description in descriptions)
    assert all(len(description) <= 90 for description in descriptions)
