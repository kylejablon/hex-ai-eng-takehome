"""Tool for listing the tables available within a database schema."""

from pydantic import BaseModel, ConfigDict, Field

from framework.agent import Tool
from framework.database import list_schemas, list_tables


class ExploreSchemaArgs(BaseModel):
    """Arguments for the explore_schema tool.

    The JSON key is "schema", but a field literally named ``schema`` shadows a
    BaseModel attribute, so we expose it under an alias and dump by alias.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(alias="schema")
    grep: str | None = None


def explore_schema(schema: str, grep: str | None = None) -> str:
    """List the tables in a schema, optionally filtered by a substring.

    Use this first to discover which tables live in a schema, then use
    explore_table to inspect a table's columns and sample data.

    Args:
        schema: Name of the schema to explore (e.g., "financial"). Case-sensitive.
        grep: Optional case-insensitive substring; only tables whose name
            contains it are returned.

    Returns:
        A newline-separated list of table names, or a helpful message listing
        the available schemas if the schema is not found.
    """
    tables = list_tables(schema)

    if not tables:
        # Unknown schema (or genuinely empty): point the agent at the real
        # schema names so it can self-correct (schema names are case-sensitive).
        available = ", ".join(list_schemas())
        return (
            f"No tables found in schema '{schema}'. "
            f"Available schemas: {available}"
        )

    if grep is not None:
        needle = grep.lower()
        matched = [t for t in tables if needle in t.lower()]
        if not matched:
            return f"No tables in schema '{schema}' match grep '{grep}'."
        tables = matched

    header = f"{len(tables)} table(s) in schema '{schema}':"
    return header + "\n" + "\n".join(tables)


EXPLORE_SCHEMA: Tool = Tool(
    name="explore_schema",
    description=(
        "List the tables available in a database schema. Use this to discover "
        "table names before inspecting them with explore_table. Schema names are "
        "case-sensitive; if the schema is not found, the available schemas are "
        "returned so you can retry."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema": {
                "type": "string",
                "description": "Name of the schema to list tables for (case-sensitive).",
            },
            "grep": {
                "type": "string",
                "description": (
                    "Optional case-insensitive substring to filter table names by."
                ),
            },
        },
        "required": ["schema"],
    },
    function=explore_schema,
    args_model=ExploreSchemaArgs,
)
