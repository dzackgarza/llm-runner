"""Low-level model invocation for llm_runner."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import ModelHTTPError

from llm_runner.contracts import ChatMessage, InvokeRequest, InvokeResponse
from llm_runner.providers import api_key, make_model, resolve
from llm_runner.schema_compiler import (
    compile_output_schema,
    validate_output_schema_instance,
)

logger = logging.getLogger(__name__)


class StructuredOutputValidationError(ValueError):
    """Raised when a model response does not satisfy the requested JSON Schema."""


@dataclass(slots=True)
class ModelInvocationResult:
    """Raw output from the chosen model invocation."""

    model: str
    output: Any


def _system_prompt(messages: list[ChatMessage]) -> str:
    """Combine all system messages into one system prompt."""
    return "\n\n".join(
        message.content for message in messages if message.role == "system"
    )


def _prompt_text(messages: list[ChatMessage]) -> str:
    """Flatten chat messages into one provider prompt."""
    parts: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            parts.append(message.content)
            continue
        parts.append(f"{message.role.upper()}:\n{message.content}")
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _translate_model_error(
    model_slug: str, env_var: str | None, error: ModelHTTPError
) -> RuntimeError:
    """Convert provider HTTP errors into readable runtime failures."""
    status = error.status_code
    if status == 429:
        return RuntimeError(
            f"Rate limit or quota exceeded for {model_slug} (HTTP 429). "
            "Wait and retry, or choose a different model."
        )
    if status in {401, 403}:
        return RuntimeError(
            f"Authentication failed for {model_slug} (HTTP {status}). "
            f"Check that {env_var} is set and valid."
        )
    if status == 400:
        return RuntimeError(f"Bad request to {model_slug} (HTTP 400): {error.body}")
    return RuntimeError(f"API error for {model_slug} (HTTP {status}): {error.body}")


async def _call_one_model(
    model_slug: str,
    request: InvokeRequest,
    *,
    output_type: Any,
) -> Any:
    """Call one specific model slug."""
    cfg, model_id = resolve(model_slug)
    key = api_key(cfg)
    if not key and cfg.env_var is not None:
        raise ValueError(f"{cfg.env_var} not set (required for {model_slug})")

    model = make_model(cfg, model_id, model_slug)
    agent: Agent[None, Any] = Agent(
        model,
        output_type=output_type,
        system_prompt=_system_prompt(request.messages),
        retries=request.options.retries,
    )
    try:
        result = await agent.run(
            _prompt_text(request.messages),
            model_settings=ModelSettings(
                temperature=request.options.temperature,
                max_tokens=request.options.max_tokens,
            ),
        )
    except ModelHTTPError as exc:
        raise _translate_model_error(model_slug, cfg.env_var, exc) from exc
    return result.output


async def _call_models(request: InvokeRequest) -> ModelInvocationResult:
    """Try the requested models in order and return the first success."""
    output_type: Any = str
    if request.output_schema is not None:
        schema_name = str(request.output_schema.get("title", "RunnerStructuredOutput"))
        output_type = compile_output_schema(request.output_schema, name=schema_name)

    last_error: Exception | None = None
    for model_slug in request.models:
        cfg, _ = resolve(model_slug)
        if api_key(cfg) == "" and cfg.env_var is not None:
            logger.debug("Skipping %s because %s is unset", model_slug, cfg.env_var)
            continue
        try:
            return ModelInvocationResult(
                model=model_slug,
                output=await _call_one_model(
                    model_slug, request, output_type=output_type
                ),
            )
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_slug, exc)
            last_error = exc
    raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error


def _structured_payload(output: Any, schema: dict[str, Any]) -> Any:
    """Coerce structured output into JSON data and validate it."""
    payload = output
    if isinstance(output, str):
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise StructuredOutputValidationError(
                "Model returned text instead of JSON for structured output."
            ) from exc
    try:
        validate_output_schema_instance(schema, payload)
    except JsonSchemaValidationError as exc:
        raise StructuredOutputValidationError(str(exc)) from exc
    return payload


async def invoke_request(request: InvokeRequest) -> InvokeResponse:
    """Execute a direct model invocation request."""
    invocation = await _call_models(request)
    if request.output_schema is None:
        if isinstance(invocation.output, str):
            return InvokeResponse(
                model=invocation.model,
                raw_text=invocation.output,
                structured=None,
            )
        raw_text = json.dumps(invocation.output)
        return InvokeResponse(
            model=invocation.model,
            raw_text=raw_text,
            structured=invocation.output,
        )

    structured = _structured_payload(invocation.output, request.output_schema)
    return InvokeResponse(
        model=invocation.model,
        raw_text=json.dumps(structured),
        structured=structured,
    )


__all__ = [
    "ModelInvocationResult",
    "StructuredOutputValidationError",
    "invoke_request",
]
