"""Tests for low-level model invocation."""

from __future__ import annotations

import asyncio
import json

import pytest

from llm_runner.contracts import ChatMessage, InvokeOptions, InvokeRequest
from llm_runner.invoke import (
    StructuredOutputValidationError,
    _structured_payload,
    invoke_request,
)
from tests.live_support import exact_echo_schema, exact_echo_user_prompt, live_model_slug


def test_invoke_request_validates_structured_output() -> None:
    model_slug = live_model_slug()
    passphrase = "PASS_INVOKE_20260310"
    number = 7

    response = asyncio.run(
        invoke_request(
            InvokeRequest(
                models=[model_slug],
                messages=[
                    ChatMessage(
                        role="system",
                        content="Return only the requested JSON fields.",
                    ),
                    ChatMessage(
                        role="user",
                        content=exact_echo_user_prompt(passphrase, number),
                    ),
                ],
                output_schema=exact_echo_schema(),
                options=InvokeOptions(temperature=0.0, max_tokens=80, retries=2),
            )
        )
    )

    assert response.model == model_slug
    assert response.structured == {"passphrase": passphrase, "number": number}
    assert json.loads(response.raw_text) == response.structured


def test_structured_payload_rejects_schema_mismatch() -> None:
    with pytest.raises(StructuredOutputValidationError):
        _structured_payload(
            {"passphrase": "PASS_INCOMPLETE_20260310"},
            exact_echo_schema(),
        )
