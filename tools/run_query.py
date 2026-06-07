"""Tool for running a SQL query against the database for exploration/verification.

This is the agent's primary mechanism for interacting with the database. Unlike
submit_answer, it does NOT finalize the run -- it executes the query and returns
the results so the agent can verify them before submitting.
"""

import polars as pl
from pydantic import BaseModel

from framework.agent import Tool
from framework.database import execute_query

# Cap how many rows we echo back so a large result doesn't flood the context.
MAX_ROWS = 10


class RunQueryArgs(BaseModel):
    """Arguments for the run_query tool."""

    query: str


def run_query(query: str) -> str:
    """Execute a SQL query and return its results, for debugging and verification.

    This does NOT submit an answer. Use it to explore the data and confirm a
    query produces the results you expect, then call submit_answer separately
    with your final query.

    Args:
        query: The SQL query to execute. Use schema-qualified table names
            (e.g., "SELECT * FROM financial.account").

    Returns:
        The query (echoed back), the number of rows returned, and a neatly
        printed preview of the results, or the database error message if the
        query failed.
    """
    result = execute_query(query)
    if not result.is_success:
        return f"Query failed: {result.error_message}"

    # A successful query always yields a dataframe; coerce the (typing-only)
    # None case to an empty frame so it renders as "0 rows returned".
    df = result.dataframe if result.dataframe is not None else pl.DataFrame()
    total_rows = df.height
    preview = df.head(MAX_ROWS)

    truncated = total_rows > MAX_ROWS
    note = f" (showing first {MAX_ROWS})" if truncated else ""

    with pl.Config(tbl_rows=MAX_ROWS, tbl_cols=-1):
        table = str(preview)

    return (
        f"SQL:\n{query}\n\n"
        f"{total_rows} rows returned{note}:\n{table}"
    )


RUN_QUERY: Tool = Tool(
    name="run_query",
    description=(
        "Run a SQL query against the database and see its results. This is for "
        "exploration and verification -- it does NOT submit your answer. Use it to "
        "check that a query returns what you expect, then call submit_answer with "
        "your final query once the results look correct."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The SQL query to execute. Use schema-qualified table names "
                    "(e.g., 'SELECT * FROM financial.account LIMIT 10')."
                ),
            },
        },
        "required": ["query"],
    },
    function=run_query,
    args_model=RunQueryArgs,
)
