"""Tool for listing the domain reference guides available to the agent."""

import os
from pathlib import Path

from pydantic import BaseModel

from framework.agent import Tool


class ListGuidesArgs(BaseModel):
    """Arguments for the list_guides tool."""

    grep: str | None = None


# The guides live alongside the evaluation data. Resolve relative to this file
# (not the current working directory) so the tool works regardless of where the
# agent is launched from.
GUIDES_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "guides"


def list_guides(grep: str | None = None) -> str:
    """List the domain reference guides available to read.

    Each guide is a markdown document describing the business rules and analytics
    conventions for one domain (e.g., how a metric is defined, what to include or
    exclude). Call this first to find the relevant guide, then read it with
    open_guide before finalizing metric definitions.

    Args:
        grep: Optional case-insensitive substring. When provided, only filenames
            containing this substring are returned.

    Returns:
        A newline-separated list of guide filenames, or a message if none match.
    """
    try:
        names = sorted(os.listdir(GUIDES_DIR))
    except OSError as e:
        return f"Could not list guides at '{GUIDES_DIR}': {e}"

    if grep:
        needle = grep.lower()
        names = [name for name in names if needle in name.lower()]

    if not names:
        suffix = f" matching '{grep}'" if grep else ""
        return f"No guides found{suffix}."

    header = f"{len(names)} guide(s)" + (f" matching '{grep}'" if grep else "") + ":"
    return header + "\n" + "\n".join(names)


LIST_GUIDES: Tool = Tool(
    name="list_guides",
    description=(
        "List the available domain reference guides. Each guide documents the "
        "business rules and analytics conventions for one domain (metric "
        "definitions, what to include/exclude, edge cases). Use this to discover "
        "which guide is relevant to the question, then read it with open_guide. "
        "Optionally pass 'grep' to filter the filenames by a substring."
    ),
    parameters={
        "type": "object",
        "properties": {
            "grep": {
                "type": "string",
                "description": (
                    "Optional case-insensitive substring used to filter the list "
                    "to filenames containing it (e.g., 'hotel')."
                ),
            },
        },
        "required": [],
    },
    function=list_guides,
    args_model=ListGuidesArgs,
)
