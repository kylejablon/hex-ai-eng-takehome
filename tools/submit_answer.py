"""Tool for submitting a final SQL query answer to the evaluation pipeline."""

from pydantic import BaseModel

from framework.agent import ANSWER_SUBMITTED_PREFIX, Tool


class SubmitAnswerArgs(BaseModel):
    """Arguments for the submit_answer tool."""

    query: str


def submit_answer(query: str) -> str:
    """Submit a SQL query as the final answer to the evaluation.

    This tool signals that the agent has completed its reasoning and is
    ready to submit a final answer.

    Args:
        query: The SQL query to submit as the final answer.

    Returns:
        A confirmation message indicating that the submission has been received.
    """
    # Return the prefix followed by the query. The prefix signals the agent to stop.
    # IMPORTANT: Do not add any text after the query - the extraction logic in the evals
    # takes everything after the prefix as the query.
    return f"{ANSWER_SUBMITTED_PREFIX}{query}"


SUBMIT_ANSWER: Tool = Tool(
    name="submit_answer",
    description=(
        "Submit your final SQL query as the answer to the current question. "
        "Use this tool when you are confident that your query correctly "
        "answers the question."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The final SQL query to submit as your answer. "
                    "Must be valid SQL syntax with table names qualified "
                    "by schema names (e.g., 'SELECT * FROM schema.table')."
                ),
            },
        },
        "required": ["query"],
    },
    function=submit_answer,
    args_model=SubmitAnswerArgs,
)
