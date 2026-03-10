"""Typer-based JSON command surfaces for llm_runner."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from llm_runner.cli_common import build_app
from llm_runner.contracts import ErrorDetail, ErrorResponse, InvokeRequest, RunRequest
from llm_runner.invoke import StructuredOutputValidationError, invoke_request
from llm_runner.providers import list_providers_response
from llm_runner.run_templates import run_request

RequestModelT = TypeVar("RequestModelT", bound=BaseModel)

app = build_app(help_text="Template-driven LLM execution.", no_args_is_help=True)
run_app = build_app(help_text="Execute one template-defined run request.")
invoke_app = build_app(help_text="Invoke models directly from JSON.")
providers_app = build_app(
    help_text="Inspect available providers.", no_args_is_help=True
)
providers_list_app = build_app(help_text="List providers and models as JSON.")


def _read_json_input(input_path: str | None) -> str:
    """Read raw JSON text from a file or stdin."""
    if input_path is not None:
        candidate = Path(input_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        return candidate.read_text()
    return sys.stdin.read()


def _write_json_output(output_path: str | None, payload: BaseModel) -> None:
    """Write JSON payload to stdout or a file."""
    content = payload.model_dump_json(indent=2)
    if output_path is not None:
        Path(output_path).write_text(content)
    else:
        typer.echo(content)


def _write_error(output_path: str | None, error: Exception) -> None:
    """Write a structured error payload."""
    payload = ErrorResponse(
        error=ErrorDetail(
            type=type(error).__name__,
            message=str(error),
        )
    )
    _write_json_output(output_path, payload)


def _parse_request(
    model_type: type[RequestModelT], input_path: str | None
) -> RequestModelT:
    """Parse a CLI JSON request into a typed request model."""
    raw = _read_json_input(input_path)
    try:
        return model_type.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


async def execute_run_request(request: RunRequest):
    """Execute a template-defined run request."""
    return await run_request(request)


async def execute_invoke_request(request: InvokeRequest):
    """Execute a low-level invoke request."""
    return await invoke_request(request)


def _run_command(
    request_type: type[RequestModelT],
    executor,
    input_path: str | None,
    output_path: str | None,
) -> None:
    """Execute one request/response JSON command."""
    try:
        request = _parse_request(request_type, input_path)
        response = asyncio.run(executor(request))
        _write_json_output(output_path, response)
    except (
        FileNotFoundError,
        StructuredOutputValidationError,
        ValueError,
    ) as exc:
        _write_error(output_path, exc)
        raise typer.Exit(code=1) from exc


@run_app.callback(invoke_without_command=True)
def run_command(
    input_path: Annotated[
        str | None,
        typer.Option("--input", "-i", help="Read request JSON from this file."),
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write response JSON to this file."),
    ] = None,
) -> None:
    """Execute one template-defined run."""
    _run_command(RunRequest, execute_run_request, input_path, output_path)


@invoke_app.callback(invoke_without_command=True)
def invoke_command(
    input_path: Annotated[
        str | None,
        typer.Option("--input", "-i", help="Read request JSON from this file."),
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write response JSON to this file."),
    ] = None,
) -> None:
    """Invoke one low-level model request."""
    _run_command(InvokeRequest, execute_invoke_request, input_path, output_path)


@providers_list_app.callback(invoke_without_command=True)
def providers_list_command(
    output_path: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write response JSON to this file."),
    ] = None,
) -> None:
    """List providers and their live models as JSON."""
    try:
        payload = list_providers_response()
        _write_json_output(output_path, payload)
    except ValueError as exc:
        _write_error(output_path, exc)
        raise typer.Exit(code=1) from exc


providers_app.add_typer(
    providers_list_app,
    name="list",
    help="List providers and their live model slugs as JSON.",
)
app.add_typer(run_app, name="run", help="Execute one template-defined run.")
app.add_typer(invoke_app, name="invoke", help="Invoke models directly from JSON.")
app.add_typer(providers_app, name="providers", help="Inspect available providers.")


def main() -> None:
    """Run the umbrella llm-runner CLI."""
    app()


def run_main() -> None:
    """Run the standalone llm-run console script."""
    run_app()


def invoke_main() -> None:
    """Run the standalone llm-invoke console script."""
    invoke_app()


def providers_list_main() -> None:
    """Run the standalone llm-provider-list console script."""
    providers_list_app()


if __name__ == "__main__":
    main()
