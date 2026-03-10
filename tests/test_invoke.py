"""Tests for low-level model invocation."""

from __future__ import annotations

import asyncio
import json

import pytest

from llm_runner.contracts import ChatMessage, InvokeOptions, InvokeRequest
from llm_runner.invoke import (
    ModelInvocationResult,
    StructuredOutputValidationError,
    invoke_request,
)


def _output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "tier": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["tier", "reasoning"],
        "additionalProperties": False,
    }


def test_invoke_request_validates_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_models(request: InvokeRequest) -> ModelInvocationResult:
        assert request.models == ["groq/llama-3.3-70b-versatile"]
        assert request.messages == [
            ChatMessage(role="system", content="You are precise."),
            ChatMessage(role="user", content="Classify the ticket."),
        ]
        return ModelInvocationResult(
            model="groq/llama-3.3-70b-versatile",
            output={"tier": "B", "reasoning": "uses a tool"},
        )

    monkeypatch.setattr("llm_runner.invoke._call_models", fake_call_models)

    response = asyncio.run(
        invoke_request(
            InvokeRequest(
                models=["groq/llama-3.3-70b-versatile"],
                messages=[
                    ChatMessage(role="system", content="You are precise."),
                    ChatMessage(role="user", content="Classify the ticket."),
                ],
                output_schema=_output_schema(),
                options=InvokeOptions(temperature=0.1, max_tokens=600, retries=4),
            )
        )
    )

    assert response.model == "groq/llama-3.3-70b-versatile"
    assert response.structured == {"tier": "B", "reasoning": "uses a tool"}
    assert json.loads(response.raw_text) == response.structured


def test_invoke_request_rejects_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_models(_: InvokeRequest) -> ModelInvocationResult:
        return ModelInvocationResult(
            model="groq/llama-3.3-70b-versatile",
            output={"tier": "B"},
        )

    monkeypatch.setattr("llm_runner.invoke._call_models", fake_call_models)

    with pytest.raises(StructuredOutputValidationError, match="reasoning"):
        asyncio.run(
            invoke_request(
                InvokeRequest(
                    models=["groq/llama-3.3-70b-versatile"],
                    messages=[ChatMessage(role="user", content="Classify the ticket.")],
                    output_schema=_output_schema(),
                )
            )
        )
