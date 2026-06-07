"""Unit tests for tool-call argument parsing and phantom-split stitching.

These cover the gpt-oss "phantom split" failure mode, where a single tool call is
split across two slots -- the tail of its argument JSON peeled into a second,
nameless slot. `_merge_split_tool_calls` stitches the slots back together and
pydantic (partial mode) parses the whitespace-laden result.
"""

from framework.agent import (
    _merge_split_tool_calls,
    _parse_arguments,
    _parse_tool_calls_from_api,
)
from tools.explore_schema import EXPLORE_SCHEMA, ExploreSchemaArgs
from tools.explore_table import EXPLORE_TABLE, ExploreTableArgs
from tools.run_query import RUN_QUERY, RunQueryArgs


def _fn_call(name: str, arguments: str, call_id: str = "x") -> dict:
    """Build one OpenAI-format tool-call slot."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


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


def test_merge_stitches_brace_split():
    # Real call missing its closing brace; phantom carries the orphaned "}".
    payload = [
        _fn_call("explore_table", '{\n  "table_name": "a.b",\n  "limit": 5\n'),
        _fn_call("", "}", call_id=""),
    ]
    merged = _merge_split_tool_calls(payload)
    assert len(merged) == 1
    assert merged[0]["function"]["arguments"] == '{\n  "table_name": "a.b",\n  "limit": 5\n}'


def test_merge_stitches_mid_string_split():
    # The damaging variant: the split lands mid-value, so the real slot alone is
    # unparseable. Stitching recovers the full required argument.
    payload = [
        _fn_call("explore_table", '{\n  "table_name": "financial.account'),
        _fn_call("", '"\n}', call_id=""),
    ]
    merged = _merge_split_tool_calls(payload)
    assert len(merged) == 1
    args, error = _parse_arguments(merged[0]["function"]["arguments"], ExploreTableArgs)
    assert error is None
    assert args["table_name"] == "financial.account"


def test_merge_does_not_touch_input():
    payload = [
        _fn_call("explore_table", '{"table_name": "a.b"'),
        _fn_call("", '}', call_id=""),
    ]
    _merge_split_tool_calls(payload)
    # Original dicts are untouched (we copied before stitching).
    assert payload[0]["function"]["arguments"] == '{"table_name": "a.b"'


def test_merge_preserves_parallel_named_calls():
    # Two genuine named calls in one turn must NOT be merged.
    payload = [
        _fn_call("explore_schema", '{"schema": "financial"}'),
        _fn_call("run_query", '{"query": "SELECT 1"}'),
    ]
    merged = _merge_split_tool_calls(payload)
    assert [m["function"]["name"] for m in merged] == ["explore_schema", "run_query"]


def test_merge_passes_through_leading_nameless_call():
    # A nameless call with nothing before it has nothing to stitch onto.
    payload = [_fn_call("", "}", call_id="")]
    merged = _merge_split_tool_calls(payload)
    assert len(merged) == 1
    assert merged[0]["function"]["name"] == ""


def test_phantom_split_end_to_end():
    # Full pipeline: merge then parse. The mid-value split now yields ONE clean
    # call instead of a failing real call plus an ignored phantom.
    payload = [
        _fn_call("explore_table", '{\n  "table_name": "financial.account",\n  "limit": 5'),
        _fn_call("", "\n}", call_id=""),
    ]
    tools = {EXPLORE_TABLE.name: EXPLORE_TABLE, RUN_QUERY.name: RUN_QUERY}
    calls = _parse_tool_calls_from_api(_merge_split_tool_calls(payload), tools)

    assert len(calls) == 1
    assert calls[0].name == "explore_table"
    assert calls[0].error is None
    assert calls[0].arguments["table_name"] == "financial.account"
    assert calls[0].arguments["limit"] == 5


def test_optional_field_split_is_recovered():
    # Split mid-`grep` value: previously dropped silently (call ran unfiltered);
    # stitching restores the intended grep.
    payload = [
        _fn_call("explore_schema", '{\n  "schema": "financial",\n  "grep": "'),
        _fn_call("", 'provider"\n}', call_id=""),
    ]
    tools = {EXPLORE_SCHEMA.name: EXPLORE_SCHEMA}
    calls = _parse_tool_calls_from_api(_merge_split_tool_calls(payload), tools)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].arguments == {"schema": "financial", "grep": "provider"}


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
