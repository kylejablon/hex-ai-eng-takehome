"""Tool for reading a domain reference guide, optionally filtered by a substring."""

import os
from pathlib import Path

from pydantic import BaseModel

from framework.agent import Tool
from tools.list_guides import GUIDES_DIR


class OpenGuideArgs(BaseModel):
    """Arguments for the open_guide tool."""

    filename: str
    grep: str | None = None


def _resolve_guide_path(filename: str) -> Path | None:
    """Resolve a user-supplied filename to a guide path inside GUIDES_DIR.

    Strips any directory components (so '../foo' can't escape the guides dir) and
    appends a '.md' extension if missing. Returns None if the file does not exist.
    """
    base = os.path.basename(filename.strip())
    if not base:
        return None
    if not base.endswith(".md"):
        base += ".md"
    path = GUIDES_DIR / base
    if not path.is_file():
        return None
    return path


def open_guide(filename: str, grep: str | None = None) -> str:
    """Read a domain reference guide, optionally filtering to matching lines.

    Args:
        filename: The guide filename (e.g., "hotel_reservations" or
            "hotel_reservations.md"). The ".md" extension is optional.
        grep: Optional case-insensitive substring. When provided, only lines
            containing it are returned.

    Returns:
        The guide text (or the matching lines), or an error message if the guide
        could not be found.
    """
    path = _resolve_guide_path(filename)
    if path is None:
        try:
            available = sorted(os.listdir(GUIDES_DIR))
        except OSError:
            available = []
        hint = "\nAvailable guides: " + ", ".join(available) if available else ""
        return f"Guide '{filename}' not found. Use list_guides to list them.{hint}"

    text = path.read_text()

    if not grep:
        return f"{path.name}:\n{text}"

    needle = grep.lower()
    matched = [line for line in text.splitlines() if needle in line.lower()]
    if not matched:
        return f"No lines in '{path.name}' matching '{grep}'."

    return f"{path.name} (lines matching '{grep}'):\n" + "\n".join(matched)


OPEN_GUIDE: Tool = Tool(
    name="open_guide",
    description=(
        "Read a domain reference guide by filename (the '.md' extension is "
        "optional). Use list_guides first to find the right filename. Pass "
        "'grep' to return only lines containing a substring."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "The guide filename, e.g., 'hotel_reservations' or "
                    "'hotel_reservations.md'."
                ),
            },
            "grep": {
                "type": "string",
                "description": (
                    "Optional case-insensitive substring; only lines containing "
                    "it are returned."
                ),
            },
        },
        "required": ["filename"],
    },
    function=open_guide,
    args_model=OpenGuideArgs,
)
