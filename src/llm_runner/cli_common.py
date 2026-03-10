"""Shared Typer configuration for llm_runner command surfaces."""

from __future__ import annotations

import typer


CONTEXT_SETTINGS: dict[str, list[str]] = {"help_option_names": ["-h", "--help"]}


def build_app(*, help_text: str, no_args_is_help: bool = False) -> typer.Typer:
    """Create a Typer app with the shared CLI defaults for this repo."""
    return typer.Typer(
        help=help_text,
        add_completion=False,
        no_args_is_help=no_args_is_help,
        pretty_exceptions_enable=False,
        context_settings=CONTEXT_SETTINGS,
    )
