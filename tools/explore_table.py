"""Tool for inspecting a table's schema and a few sample rows."""

import polars as pl
from pydantic import BaseModel

from framework.agent import Tool
from framework.database import describe_table, execute_query


class ExploreTableArgs(BaseModel):
    """Arguments for the explore_table tool."""

    table_name: str
    columns: list[str] | None = None
    limit: int = 3


def explore_table(
    table_name: str,
    columns: list[str] | None = None,
    limit: int = 3,
) -> str:
    """Inspect a table: show its columns (with types) and a few sample rows.

    Use this after explore_schema to understand what a table contains before
    writing a query against it.

    Args:
        table_name: Fully-qualified table name as "schema.table"
            (e.g., "financial.account").
        columns: Optional list of column names to sample. Defaults to all
            columns. Prefer selecting only the columns you need.
        limit: Number of sample rows to return (default 3).

    Returns:
        A description of the table's columns and a sample of its rows, or an
        error message if the table could not be read.
    """
    schema, _, table = table_name.partition(".")
    if not table:
        return (
            f"Invalid table_name '{table_name}'. "
            "Use the fully-qualified 'schema.table' format (e.g., 'financial.account')."
        )

    column_info = describe_table(schema, table)
    if not column_info:
        return (
            f"Could not find columns for table '{table_name}'. "
            "Check the schema and table name (both are case-sensitive)."
        )

    select_cols = ", ".join(columns) if columns else "*"
    result = execute_query(f"SELECT {select_cols} FROM {table_name} LIMIT {limit}")
    if not result.is_success:
        return f"Error reading sample rows from '{table_name}': {result.error_message}"

    # Show all columns rather than letting Polars truncate them in its repr.
    with pl.Config(tbl_rows=limit, tbl_cols=-1):
        sample = str(result.dataframe)

    return (
        f"Table: {table_name}\n"
        f"Columns ({len(column_info)}):\n  " + "\n  ".join(column_info) + "\n\n"
        f"Sample rows (limit {limit}):\n{sample}"
    )


EXPLORE_TABLE: Tool = Tool(
    name="explore_table",
    description=(
        "Inspect a single table: returns its column names and types plus a few "
        "sample rows so you can understand what the data looks like. Provide the "
        "table as a fully-qualified 'schema.table' name. Optionally pass only the "
        "columns you care about to keep the output focused."
    ),
    parameters={
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": (
                    "Fully-qualified table name as 'schema.table' "
                    "(e.g., 'financial.account')."
                ),
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional subset of column names to sample. Defaults to all columns."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Number of sample rows to return (default 3).",
            },
        },
        "required": ["table_name"],
    },
    function=explore_table,
    args_model=ExploreTableArgs,
)
