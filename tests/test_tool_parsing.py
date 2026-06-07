"""Unit tests for tool-call argument parsing (pydantic partial JSON).

These focus on the gpt-oss "phantom split" failure mode, where a single tool
call's closing brace is peeled off into a second, nameless tool-call slot --
leaving the real call's argument JSON truncated and previously unparseable.
"""

from framework.agent import _parse_arguments, _parse_tool_calls_from_api
from tools.explore_schema import ExploreSchemaArgs
from tools.explore_table import EXPLORE_TABLE, ExploreTableArgs
from tools.run_query import RUN_QUERY, RunQueryArgs


def test_complete_arguments_parse():
    args, error = _parse_arguments('{"table_name": "a.b", "limit": 2}', ExploreTableArgs)
    assert error is None
    assert args == {"table_name": "a.b", "columns": None, "limit": 2}


def test_truncated_arguments_recovered_via_partial():
    # The real half of a phantom split: valid JSON missing its closing brace.
    raw = '{\n  "table_name": "financial.account",\n  "limit": 5\n'
    args, error = _parse_arguments(raw, ExploreTableArgs)
    assert error is None
    assert args["table_name"] == "financial.account"
    assert args["limit"] == 5
    # A field truncated away falls back to its default rather than erroring.
    assert args["columns"] is None


def test_missing_required_field_is_error():
    args, error = _parse_arguments('{"limit": 5}', ExploreTableArgs)
    assert error is not None
    assert args == {}


def test_schema_field_alias_dumps_to_keyword():
    # JSON key is "schema" (aliased to schema_name); it must dump back to "schema"
    # so explore_schema(**args) receives the right keyword.
    args, error = _parse_arguments('{"schema": "financial"}', ExploreSchemaArgs)
    assert error is None
    assert args == {"schema": "financial", "grep": None}


def test_phantom_split_payload_end_to_end():
    # Exactly the shape seen in the eval traces: a real call missing its brace,
    # plus a nameless phantom call carrying the orphaned "}".
    payload = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "explore_table",
                "arguments": '{\n  "table_name": "financial.account",\n  "limit": 5\n',
            },
        },
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": "}"},
        },
    ]
    tools = {EXPLORE_TABLE.name: EXPLORE_TABLE, RUN_QUERY.name: RUN_QUERY}
    calls = _parse_tool_calls_from_api(payload, tools)

    assert len(calls) == 2
    real, phantom = calls
    # The real call now parses cleanly instead of failing on invalid JSON.
    assert real.name == "explore_table"
    assert real.error is None
    assert real.arguments["table_name"] == "financial.account"
    assert real.arguments["limit"] == 5
    # The phantom call maps to no tool and carries no arguments to act on.
    assert phantom.name == ""
    assert phantom.arguments == {}


def test_unknown_tool_is_passed_through_without_args():
    payload = [
        {"id": "x", "type": "function", "function": {"name": "nope", "arguments": "{}"}}
    ]
    calls = _parse_tool_calls_from_api(payload, {RUN_QUERY.name: RUN_QUERY})
    assert len(calls) == 1
    assert calls[0].name == "nope"
    assert calls[0].arguments == {}


def test_models_mirror_function_signatures():
    # Defaults declared on the models match the tool function defaults.
    assert ExploreTableArgs(table_name="t").model_dump() == {
        "table_name": "t",
        "columns": None,
        "limit": 3,
    }
    assert RunQueryArgs(query="SELECT 1").model_dump() == {"query": "SELECT 1"}
