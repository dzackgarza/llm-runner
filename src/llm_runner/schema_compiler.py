"""JSON Schema helpers for llm_runner."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic_ai import StructuredDict


def compile_output_schema(
    schema: dict[str, Any],
    *,
    name: str = "RunnerStructuredOutput",
    description: str | None = None,
) -> type[dict[str, Any]]:
    """Compile JSON Schema into a pydantic-ai structured-output type."""
    Draft202012Validator.check_schema(schema)
    return StructuredDict(schema, name=name, description=description)


def validate_output_schema_instance(schema: dict[str, Any], payload: Any) -> None:
    """Validate one structured payload against JSON Schema."""
    Draft202012Validator(schema=schema).validate(payload)


__all__ = [
    "JsonSchemaValidationError",
    "compile_output_schema",
    "validate_output_schema_instance",
]
