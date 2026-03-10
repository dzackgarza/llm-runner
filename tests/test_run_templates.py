"""Tests for template-defined runs."""

from __future__ import annotations

import asyncio

from llm_runner.contracts import RunRequest
from llm_runner.run_templates import run_request
from tests.live_support import (
    DIFF_TEXT,
    SYSTEM_PROMPT,
    live_model_slug,
    write_exact_echo_templates,
)


def test_run_request_renders_input_and_response_templates(tmp_path) -> None:
    model_slug = live_model_slug()
    passphrase = "PASS_RUN_TEMPLATE_20260310"
    number = 23
    run_template, diff_file = write_exact_echo_templates(tmp_path, model_slug=model_slug)
    response = asyncio.run(
        run_request(
            RunRequest(
                template={"path": str(run_template)},
                bindings={
                    "data": {"passphrase": passphrase, "number": number},
                    "text_files": [{"name": "diff", "path": str(diff_file)}],
                },
            )
        )
    )

    assert response.run.template_path == str(run_template.resolve())
    assert response.run.model == model_slug
    assert response.run.messages[0].role == "system"
    assert response.run.messages[0].content == SYSTEM_PROMPT
    assert response.run.messages[1].role == "user"
    assert f"Use passphrase {passphrase} and number {number}." in response.run.messages[1].content
    assert DIFF_TEXT in response.run.messages[1].content
    assert response.response.structured == {
        "passphrase": passphrase,
        "number": number,
    }
    assert response.final_output.text is None
    assert response.final_output.data == {
        "passphrase": passphrase,
        "number": number,
        "diff_present": True,
    }
