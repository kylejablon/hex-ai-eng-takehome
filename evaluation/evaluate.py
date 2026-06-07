#!/usr/bin/env python3
"""Evaluation script for your agent.

Please do not modify this file in your submission, except for adding any tools you write
to the create_tools function.

This script runs the agent against evaluation datasets and reports results
with a visual progress indicator.

Usage:
    uv run evaluate --api-key YOUR_API_KEY --concurrency 16
    uv run evaluate --api-key YOUR_API_KEY --split easy
    uv run evaluate --api-key YOUR_API_KEY --split both

You can also run the script with the --verbose flag to see detailed progress.

The --split flag controls which evaluation set to run:
    - "easy": Run only the easy evaluation cases
    - "hard": Run only the hard evaluation cases (default)
    - "both": Run both easy and hard evaluation cases

To customize the tools available to the agent, modify the `create_tools()`
function below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import polars as pl
import sqlglot
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from evaluation.compare import loosely_compare_dataframes
from framework.agent import ANSWER_SUBMITTED_PREFIX, Agent, AgentEvent, EventType, Tool
from framework.database import execute_query
from framework.llm import OpenRouterConfig, TokenUsage
from tools.explore_schema import EXPLORE_SCHEMA
from tools.explore_table import EXPLORE_TABLE
from tools.list_guides import LIST_GUIDES
from tools.open_guide import OPEN_GUIDE
from tools.run_query import RUN_QUERY
from tools.submit_answer import SUBMIT_ANSWER

# =============================================================================
# Evaluation Configuration
# =============================================================================


@dataclass
class EvalConfig:
    """Configuration for evaluation runs.

    This replaces global variables for verbose logging and trace logging,
    making the code more testable and thread-safe when passed explicitly.
    """

    verbose: bool = False
    log_dir: Path | None = None

    def log_verbose(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Convert an AgentEvent to a serializable dictionary."""
    return {
        "type": event.type.name,
        "data": event.data,
    }


def save_trace(
    case: EvalCase,
    events: list[AgentEvent],
    result: EvalResult,
    trace_id: str,
    log_dir: Path,
    duration_seconds: float | None = None,
) -> Path:
    """Save the full conversation trace to a JSON file.

    Args:
        case: The evaluation case that was run.
        events: List of all agent events from the run.
        result: The evaluation result.
        trace_id: Unique identifier for this trace.
        log_dir: Directory to save the trace to.
        duration_seconds: How long the eval took in seconds.

    Returns:
        Path to the saved trace file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "trace_id": trace_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_seconds": duration_seconds,
        "case": {
            "prompt": case.prompt,
            "gold_query": case.gold_query,
        },
        "events": [_event_to_dict(e) for e in events],
        "result": {
            "passed": result.passed,
            "submitted_query": result.submitted_query,
            "error": result.error,
            "failure_type": result.failure_type.name,
        },
    }

    trace_file = log_dir / f"{trace_id}.json"
    with open(trace_file, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)

    return trace_file


# =============================================================================
# Configuration - Modify this function to customize agent tools
# =============================================================================


def create_tools() -> dict[str, Tool]:
    """Create the tools for the agent.

    Modify this function to add or remove tools from the agent.
    The agent will have access to all tools returned by this function.

    Returns:
        A dictionary mapping tool names to Tool objects.
    """
    return {
        SUBMIT_ANSWER.name: SUBMIT_ANSWER,
        EXPLORE_SCHEMA.name: EXPLORE_SCHEMA,
        EXPLORE_TABLE.name: EXPLORE_TABLE,
        RUN_QUERY.name: RUN_QUERY,
        LIST_GUIDES.name: LIST_GUIDES,
        OPEN_GUIDE.name: OPEN_GUIDE,
    }


# =============================================================================
# Evaluation Types and Logic
# =============================================================================


class FailureType(Enum):
    """Classification of evaluation failure types.

    Used to distinguish between "real" failures (agent submitted wrong answer)
    and infrastructure/error failures.
    """

    # Not a failure - evaluation passed
    NONE = auto()
    # Real mismatch - agent submitted an answer but it was wrong
    MISMATCH = auto()
    # Agent didn't submit an answer
    NO_SUBMISSION = auto()
    # Agent encountered an error during execution
    AGENT_ERROR = auto()
    # Submitted SQL query failed to execute
    SQL_ERROR = auto()
    # Infrastructure error (gold query failed, unexpected None, etc.)
    INFRA_ERROR = auto()
    # Unexpected exception during evaluation
    EXCEPTION = auto()


@dataclass
class EvalCase:
    """A single evaluation case from the eval dataset."""

    prompt: str
    gold_query: str


@dataclass
class EvalResult:
    """Result of running a single evaluation."""

    case: EvalCase
    submitted_query: str | None
    passed: bool
    error: str | None = None
    failure_type: FailureType = FailureType.NONE
    # Store dataframes for failed comparisons to enable detailed debugging
    gold_df: pl.DataFrame | None = None
    submitted_df: pl.DataFrame | None = None
    # Token usage for this evaluation
    usage: TokenUsage | None = None


@dataclass
class EvalSplitResults:
    """Results for an entire evaluation split (e.g., easy or hard)."""

    name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def failed_mismatch(self) -> int:
        """Count of failures due to actual dataframe mismatch."""
        return sum(1 for r in self.results if r.failure_type == FailureType.MISMATCH)

    @property
    def failed_other(self) -> int:
        """Count of failures due to errors/infra issues (not mismatches)."""
        return sum(
            1
            for r in self.results
            if not r.passed and r.failure_type != FailureType.MISMATCH
        )

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def total_usage(self) -> TokenUsage:
        """Total token usage across all results in this split."""
        total = TokenUsage()
        for r in self.results:
            if r.usage is not None:
                total = total + r.usage
        return total


def load_eval_cases(eval_file: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file."""
    with open(eval_file) as f:
        data = json.load(f)
    return [EvalCase(prompt=item["prompt"], gold_query=item["query"]) for item in data]


def extract_submitted_answer_from_events(
    agent: Agent,
    case: EvalCase,
    config: EvalConfig,
) -> tuple[str | None, str | None, list[AgentEvent], TokenUsage | None]:
    """Run the agent and extract the submitted answer from the event stream.

    This function processes agent events to capture the submitted SQL query
    without relying on global state. The submitted query is extracted directly
    from the tool result, which contains the query after the ANSWER_SUBMITTED_PREFIX.

    Args:
        agent: The agent to run.
        case: The evaluation case containing the prompt.
        config: Evaluation configuration for logging.

    Returns:
        A tuple of (submitted_query, error_message, events_list, token_usage).
        If successful, submitted_query contains the SQL and error_message is None.
        If failed, submitted_query may be None and error_message describes
        the issue. events_list contains all events from the run for logging.
        token_usage contains the total tokens used during the run.
    """
    submitted_query: str | None = None
    events: list[AgentEvent] = []
    usage: TokenUsage | None = None

    for event in agent.run(case.prompt):
        events.append(event)

        # Verbose logging for key events
        if event.type == EventType.ITERATION_START:
            config.log_verbose(f"    Iteration {event.data.get('iteration', '?')} starting...")
        elif event.type == EventType.GENERATION_START:
            config.log_verbose("    Waiting for LLM response...")
        elif event.type == EventType.GENERATION_END:
            config.log_verbose("    LLM response received")
        elif event.type == EventType.TOOL_EXECUTION_START:
            tool_name = event.data.get("name", "unknown")
            config.log_verbose(f"    Executing tool: {tool_name}")
        elif event.type == EventType.TOOL_EXECUTION_END:
            tool_name = event.data.get("name", "unknown")
            config.log_verbose(f"    Tool {tool_name} completed")

        # Check for agent errors
        if event.type == EventType.AGENT_ERROR:
            error = event.data.get("error", "Unknown")
            usage = event.data.get("usage")
            return None, f"Agent error: {error}", events, usage

        # Capture token usage from completion events
        if event.type == EventType.AGENT_COMPLETE:
            usage = event.data.get("usage")

        # When a tool execution ends, check if it was a successful submit
        if event.type == EventType.TOOL_EXECUTION_END:
            if event.data.get("name") == "submit_answer":
                result = event.data.get("result", "")
                if result.startswith(ANSWER_SUBMITTED_PREFIX):
                    # Extract the query - everything after the prefix is the query
                    submitted_query = result[len(ANSWER_SUBMITTED_PREFIX) :].strip()

    return submitted_query, None, events, usage


def run_single_eval(
    agent: Agent,
    case: EvalCase,
    config: EvalConfig,
) -> EvalResult:
    """Run a single evaluation case.

    Args:
        agent: The agent to evaluate.
        case: The evaluation case to run.
        config: Evaluation configuration for logging and tracing.

    Returns:
        An EvalResult with the outcome.
    """
    events: list[AgentEvent] = []
    trace_id = str(uuid.uuid4())
    start_time = time.monotonic()

    prompt_preview = case.prompt[:60] + "..." if len(case.prompt) > 60 else case.prompt
    config.log_verbose(f"Starting eval: {prompt_preview}")

    try:
        # Run the agent and extract the submitted answer from events
        config.log_verbose("  Running agent...")
        submitted_query, error, events, usage = extract_submitted_answer_from_events(
            agent, case, config
        )
        agent_duration = time.monotonic() - start_time
        config.log_verbose(f"  Agent finished in {agent_duration:.1f}s")

        if error is not None:
            duration = time.monotonic() - start_time
            config.log_verbose(f"  Agent error: {error}")
            result = EvalResult(
                case=case,
                submitted_query=submitted_query,
                passed=False,
                error=error,
                failure_type=FailureType.AGENT_ERROR,
                usage=usage,
            )
            _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
            return result

        if submitted_query is None:
            duration = time.monotonic() - start_time
            config.log_verbose("  No answer submitted")
            result = EvalResult(
                case=case,
                submitted_query=None,
                passed=False,
                error="No answer submitted (agent did not call submit_answer)",
                failure_type=FailureType.NO_SUBMISSION,
                usage=usage,
            )
            _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
            return result

        # Execute the gold query
        config.log_verbose("  Executing gold query...")
        gold_start = time.monotonic()
        gold_result = execute_query(case.gold_query)
        config.log_verbose(f"  Gold query took {time.monotonic() - gold_start:.1f}s")

        if not gold_result.is_success:
            duration = time.monotonic() - start_time
            config.log_verbose(f"  Gold query failed: {gold_result.error_message}")
            result = EvalResult(
                case=case,
                submitted_query=submitted_query,
                passed=False,
                error=f"Gold query execution failed: {gold_result.error_message}",
                failure_type=FailureType.INFRA_ERROR,
                usage=usage,
            )
            _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
            return result

        # Execute the submitted query
        config.log_verbose("  Executing submitted query...")
        sub_start = time.monotonic()
        submitted_result = execute_query(submitted_query)
        config.log_verbose(f"  Submitted query took {time.monotonic() - sub_start:.1f}s")

        if not submitted_result.is_success:
            duration = time.monotonic() - start_time
            config.log_verbose(f"  Submitted query failed: {submitted_result.error_message}")
            result = EvalResult(
                case=case,
                submitted_query=submitted_query,
                passed=False,
                error=f"Submitted query execution failed: {submitted_result.error_message}",
                failure_type=FailureType.SQL_ERROR,
                usage=usage,
            )
            _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
            return result

        # Compare the results
        gold_df = gold_result.dataframe
        submitted_df = submitted_result.dataframe
        if gold_df is None or submitted_df is None:
            duration = time.monotonic() - start_time
            result = EvalResult(
                case=case,
                submitted_query=submitted_query,
                passed=False,
                error="Unexpected None dataframe after successful query",
                failure_type=FailureType.INFRA_ERROR,
                usage=usage,
            )
            _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
            return result

        config.log_verbose("  Comparing results...")
        passed = loosely_compare_dataframes(gold_df, submitted_df)
        duration = time.monotonic() - start_time
        config.log_verbose(f"  Result: {'PASS' if passed else 'FAIL'} ({duration:.1f}s total)")

        result = EvalResult(
            case=case,
            submitted_query=submitted_query,
            passed=passed,
            error=None if passed else "Results do not match",
            failure_type=FailureType.NONE if passed else FailureType.MISMATCH,
            # Store dataframes for debugging when comparison fails
            gold_df=gold_df if not passed else None,
            submitted_df=submitted_df if not passed else None,
            usage=usage,
        )
        _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
        return result

    except Exception as e:
        duration = time.monotonic() - start_time
        config.log_verbose(f"  Exception: {e!s}")
        result = EvalResult(
            case=case,
            submitted_query=None,
            passed=False,
            error=f"Exception: {e!s}",
            failure_type=FailureType.EXCEPTION,
            # Note: usage may not be available if exception occurred early
        )
        _maybe_save_trace(case, events, result, trace_id, config.log_dir, duration)
        return result


def _maybe_save_trace(
    case: EvalCase,
    events: list[AgentEvent],
    result: EvalResult,
    trace_id: str,
    log_dir: Path | None,
    duration: float | None = None,
) -> None:
    """Save trace if logging is enabled.

    Args:
        case: The evaluation case that was run.
        events: List of all agent events from the run.
        result: The evaluation result.
        trace_id: Unique identifier for this trace.
        log_dir: Directory to save the trace to, or None to skip saving.
        duration: How long the eval took in seconds.
    """
    if log_dir is not None:
        save_trace(case, events, result, trace_id, log_dir, duration)


# =============================================================================
# Progress Display
# =============================================================================


def create_progress_bar(results: list[EvalResult], width: int = 50) -> Text:
    """Create a colored progress bar showing pass/fail status.

    Args:
        results: List of evaluation results.
        width: Width of the progress bar in characters.

    Returns:
        A Rich Text object with colored segments.
    """
    if not results:
        return Text("░" * width, style="dim")

    text = Text()
    for result in results:
        char = "█"
        style = "green" if result.passed else "red"
        text.append(char, style=style)

    # Pad remaining space
    remaining = width - len(results)
    if remaining > 0:
        text.append("░" * remaining, style="dim")

    return text


def create_status_table(
    split_name: str,
    results: list[EvalResult],
    total: int,
) -> Table:
    """Create a status table for display during evaluation.

    Args:
        split_name: Name of the evaluation split.
        results: Current results.
        total: Total number of evaluations.

    Returns:
        A Rich Table for display.
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="cyan", width=12)
    table.add_column("Progress", width=total + 4)
    table.add_column("Stats", width=20)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    progress_bar = create_progress_bar(results, total)
    stats = f"[green]{passed}[/green] / [red]{failed}[/red] / {total}"

    table.add_row(split_name, progress_bar, stats)

    return table


# =============================================================================
# Main Evaluation Runner
# =============================================================================


def _run_single_eval_worker(
    case: EvalCase,
    case_index: int,
    tools: dict[str, Tool],
    api_key: str,
    log_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[int, EvalResult]:
    """Worker function to run a single evaluation in a thread.

    Creates its own Agent instance to avoid shared state issues.

    Args:
        case: The evaluation case to run.
        case_index: Index of the case (for ordering results).
        tools: Tools to provide to the agent.
        api_key: OpenRouter API key.
        log_dir: Optional directory to save agent traces to.
        verbose: Whether to enable verbose logging.

    Returns:
        A tuple of (case_index, EvalResult) for proper ordering.
    """
    # Create evaluation config for this worker
    eval_config = EvalConfig(verbose=verbose, log_dir=log_dir)

    # Each worker creates its own agent to avoid shared state
    llm_config = OpenRouterConfig(api_key=api_key)
    agent = Agent(config=llm_config, tools=tools)
    result = run_single_eval(agent, case, eval_config)
    return case_index, result


def evaluate_split(
    tools: dict[str, Tool],
    eval_file: Path,
    console: Console,
    api_key: str,
    concurrency: int = 1,
    log_dir: Path | None = None,
    max_cases: int | None = None,
    verbose: bool = False,
) -> EvalSplitResults:
    """Run evaluation on a single split.

    Args:
        tools: Tools to provide to the agent.
        eval_file: Path to the evaluation JSON file.
        console: Rich console for output.
        api_key: OpenRouter API key.
        concurrency: Number of parallel evaluations to run.
        log_dir: Optional directory to save agent traces to.
        max_cases: Optional limit on the number of cases to run.
        verbose: Whether to enable verbose logging.

    Returns:
        EvalSplitResults containing all results for this split.
    """
    split_name = eval_file.stem
    cases = load_eval_cases(eval_file)

    # Limit cases if max_cases is specified
    if max_cases is not None and max_cases < len(cases):
        cases = cases[:max_cases]
        split_name = f"{split_name} (first {max_cases})"

    split_results = EvalSplitResults(name=split_name)

    # Set up logging for this split
    split_log_dir: Path | None = None
    if log_dir is not None:
        split_log_dir = log_dir / eval_file.stem  # Use original name for directory
        split_log_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Logging traces to: {split_log_dir}[/dim]")

    console.print(f"\n[bold cyan]Evaluating: {split_name}[/bold cyan]")
    console.print(f"[dim]Loaded {len(cases)} evaluation cases[/dim]")
    console.print(f"[dim]Running with concurrency: {concurrency}[/dim]\n")

    if concurrency == 1:
        # Sequential execution (original behavior)
        llm_config = OpenRouterConfig(api_key=api_key)
        agent = Agent(config=llm_config, tools=tools)
        eval_config = EvalConfig(verbose=verbose, log_dir=split_log_dir)

        with Live(
            create_status_table(split_name, [], len(cases)),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for case in cases:
                agent.reset_conversation()
                result = run_single_eval(agent, case, eval_config)
                split_results.results.append(result)
                live.update(
                    create_status_table(
                        split_name,
                        split_results.results,
                        len(cases),
                    )
                )
    else:
        # Parallel execution
        # Pre-allocate results list with None placeholders to maintain order
        results_by_index: dict[int, EvalResult] = {}
        completed_count = 0
        results_lock = threading.Lock()

        with Live(
            create_status_table(split_name, [], len(cases)),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        _run_single_eval_worker,
                        case,
                        idx,
                        tools,
                        api_key,
                        split_log_dir,
                        verbose,
                    ): idx
                    for idx, case in enumerate(cases)
                }

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        case_index, result = future.result()
                    except Exception as e:
                        # Handle unexpected errors from the worker
                        case_index = futures[future]
                        result = EvalResult(
                            case=cases[case_index],
                            submitted_query=None,
                            passed=False,
                            error=f"Worker exception: {e!s}",
                            failure_type=FailureType.EXCEPTION,
                        )

                    with results_lock:
                        results_by_index[case_index] = result
                        completed_count += 1

                        # Build ordered results list for display
                        # Show results in order up to current completion
                        ordered_results = [
                            results_by_index[i]
                            for i in range(len(cases))
                            if i in results_by_index
                        ]

                        live.update(
                            create_status_table(
                                split_name,
                                ordered_results,
                                len(cases),
                            )
                        )

        # Build final ordered results
        split_results.results = [results_by_index[i] for i in range(len(cases))]

    return split_results


def _format_sql(query: str) -> str:
    """Format a SQL query using SQLGlot for pretty printing.

    Args:
        query: The SQL query to format.

    Returns:
        A nicely formatted SQL query string, or the original if formatting fails.
    """
    try:
        return sqlglot.transpile(query, read="duckdb", pretty=True)[0]
    except Exception:
        # If formatting fails, return the original query
        return query


def _dataframe_to_table(df: pl.DataFrame, title: str, max_rows: int = 20) -> Table:
    """Convert a Polars DataFrame to a Rich Table.

    Args:
        df: The DataFrame to convert.
        title: Title for the table.
        max_rows: Maximum number of rows to display.

    Returns:
        A Rich Table representation of the DataFrame.
    """
    table = Table(title=title, show_header=True, header_style="bold", expand=True)

    # Add columns
    for col_name in df.columns:
        table.add_column(col_name, overflow="fold")

    # Add rows (limit to max_rows)
    rows_to_show = min(df.height, max_rows)
    for i in range(rows_to_show):
        row_values = [str(df[col][i]) for col in df.columns]
        table.add_row(*row_values)

    if df.height > max_rows:
        table.add_row(*[f"... ({df.height - max_rows} more)" for _ in df.columns])

    return table


def render_comparison_failure(
    result: EvalResult,
    console: Console,
    max_rows: int = 20,
) -> None:
    """Render a side-by-side comparison of gold and submitted dataframes.

    Args:
        result: The failed EvalResult containing the dataframes.
        console: Rich console for output.
        max_rows: Maximum number of rows to display per dataframe.
    """
    # Early return if dataframes are not available
    gold_df = result.gold_df
    submitted_df = result.submitted_df
    if gold_df is None or submitted_df is None:
        return

    # Create a container table for side-by-side layout
    comparison_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
    comparison_table.add_column("Gold", ratio=1)
    comparison_table.add_column("Submitted", ratio=1)

    # Create the dataframe tables
    gold_table = _dataframe_to_table(
        gold_df,
        f"[green]Gold[/green] ({gold_df.height} rows, {gold_df.width} cols)",
        max_rows,
    )
    submitted_title = (
        f"[red]Submitted[/red] "
        f"({submitted_df.height} rows, {submitted_df.width} cols)"
    )
    submitted_table = _dataframe_to_table(
        submitted_df,
        submitted_title,
        max_rows,
    )

    comparison_table.add_row(gold_table, submitted_table)

    # Format and print the queries (without boxes)
    gold_query = _format_sql(result.case.gold_query)
    submitted_query_raw = result.submitted_query
    if submitted_query_raw is not None:
        submitted_query = _format_sql(submitted_query_raw)
    else:
        submitted_query = "(no query submitted)"

    console.print("[green]Gold Query:[/green]")
    console.print(gold_query)
    console.print()
    console.print("[red]Submitted Query:[/red]")
    console.print(submitted_query)
    console.print()

    # Print the dataframes
    console.print(comparison_table)
    console.print()


def print_summary(
    all_results: list[EvalSplitResults],
    console: Console,
    *,
    verbose: bool = False,
) -> None:
    """Print a summary of all evaluation results.

    Args:
        all_results: Results from all evaluation splits.
        console: Rich console for output.
        verbose: If True, show detailed failure information for each failed case.
    """
    console.print("\n" + "=" * 60)
    console.print("[bold]Evaluation Summary[/bold]")
    console.print("=" * 60 + "\n")

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Mismatch", justify="right", style="yellow")
    table.add_column("Other", justify="right", style="magenta")
    table.add_column("Total", justify="right")
    table.add_column("Pass Rate", justify="right")

    total_passed: int = 0
    total_mismatch: int = 0
    total_other: int = 0
    total_count: int = 0
    total_usage = TokenUsage()

    for split in all_results:
        pass_rate = f"{split.pass_rate:.1%}"
        style = "green" if split.pass_rate >= 0.8 else "yellow" if split.pass_rate >= 0.5 else "red"
        table.add_row(
            split.name,
            str(split.passed),
            str(split.failed_mismatch),
            str(split.failed_other),
            str(split.total),
            f"[{style}]{pass_rate}[/{style}]",
        )
        total_passed += split.passed
        total_mismatch += split.failed_mismatch
        total_other += split.failed_other
        total_count += split.total
        total_usage = total_usage + split.total_usage

    # Add total row
    if len(all_results) > 1 and total_count > 0:
        table.add_section()
        # Cast to float to satisfy type checker
        overall_rate = float(total_passed) / float(total_count)
        style = "green" if overall_rate >= 0.8 else "yellow" if overall_rate >= 0.5 else "red"
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_passed}[/bold]",
            f"[bold]{total_mismatch}[/bold]",
            f"[bold]{total_other}[/bold]",
            f"[bold]{total_count}[/bold]",
            f"[bold {style}]{overall_rate:.1%}[/bold {style}]",
        )

    console.print(table)

    # Print token usage summary
    console.print()
    console.print("[bold]Token Usage[/bold]")
    console.print(
        f"  Input tokens:  [cyan]{total_usage.prompt_tokens:,}[/cyan]"
    )
    console.print(
        f"  Output tokens: [cyan]{total_usage.completion_tokens:,}[/cyan]"
    )
    console.print(
        f"  Total tokens:  [cyan]{total_usage.total_tokens:,}[/cyan]"
    )

    # Show failed cases details only in verbose mode
    if verbose:
        for split in all_results:
            failed_results = [r for r in split.results if not r.passed]
            if failed_results:
                console.print(f"\n[bold red]Failed cases in {split.name}:[/bold red]")
                for i, result in enumerate(failed_results):
                    prompt = result.case.prompt
                    console.print(f"\n  [bold]{i + 1}. {prompt}[/bold]")
                    console.print(f"     [dim]Error: {result.error}[/dim]")

                    # Render side-by-side comparison if dataframes are available
                    if result.gold_df is not None and result.submitted_df is not None:
                        console.print()
                        render_comparison_failure(result, console)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluation script for the SQL agent"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel evaluations to run (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing detailed progress for each eval.",
    )
    parser.add_argument(
        "--split",
        choices=["easy", "hard", "both"],
        default="hard",
        help="Which evaluation split to run: 'easy', 'hard', or 'both' (default: hard)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the evaluation script."""
    load_dotenv()
    args = parse_args()
    console = Console()

    console.print("[bold]SQL Agent Evaluation[/bold]")
    console.print("[dim]Loading tools and preparing agent...[/dim]\n")

    # Get tools from the configurable function
    tools = create_tools()
    console.print(f"[dim]Agent tools: {', '.join(tools.keys())}[/dim]")
    if args.concurrency > 1:
        console.print(f"[dim]Concurrency: {args.concurrency}[/dim]")

    # Set up logging directory with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Saving traces to: {log_dir}[/dim]")

    # Find evaluation files based on split argument
    data_dir = Path(__file__).parent / "data"

    if args.split == "easy":
        eval_files = [data_dir / "evals_easy.json"]
    elif args.split == "hard":
        eval_files = [data_dir / "evals_hard.json"]
    else:  # "both"
        eval_files = [
            data_dir / "evals_easy.json",
            data_dir / "evals_hard.json",
        ]
    max_cases = None

    # Filter to existing files
    eval_files = [f for f in eval_files if f.exists()]

    if not eval_files:
        console.print("[red]No evaluation files found![/red]")
        return

    # Run evaluations on each split
    all_results: list[EvalSplitResults] = []
    for eval_file in eval_files:
        try:
            results = evaluate_split(
                tools,
                eval_file,
                console,
                args.api_key,
                args.concurrency,
                log_dir,
                max_cases,
                args.verbose,
            )
            all_results.append(results)
        except KeyboardInterrupt:
            console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error evaluating {eval_file.name}: {e}[/red]")

        # Print final summary
        if all_results:
            print_summary(all_results, console, verbose=args.verbose)


if __name__ == "__main__":
    main()
