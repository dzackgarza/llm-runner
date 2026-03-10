"""Tests for template-defined runs."""

from __future__ import annotations

import asyncio
import textwrap

import pytest

from llm_runner.contracts import InvokeResponse, RunOverrides, RunRequest
from llm_runner.run_templates import run_request


def test_run_request_renders_input_and_response_templates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    system_template = prompts_dir / "system.md"
    system_template.write_text("You are precise about {{ ticket.title }}.")
    response_template = prompts_dir / "response.json.j2"
    response_template.write_text(
        textwrap.dedent(
            """
            {
              "ticket_id": {{ request.bindings.data.ticket.id }},
              "tier": {{ response.structured.tier | tojson }},
              "reasoning": {{ response.structured.reasoning | tojson }}
            }
            """
        ).strip()
    )
    run_template = prompts_dir / "classify-ticket.md"
    run_template.write_text(
        textwrap.dedent(
            """
            ---
            kind: llm-run
            models:
              - groq/llama-3.3-70b-versatile
            system_template:
              path: system.md
            temperature: 0.0
            max_tokens: 500
            retries: 3
            output_schema:
              type: object
              properties:
                tier:
                  type: string
                reasoning:
                  type: string
              required: [tier, reasoning]
              additionalProperties: false
            response_template:
              path: response.json.j2
              format: json
            ---

            Classify ticket {{ ticket.id }}.
            {{ diff }}
            """
        ).strip()
    )
    diff_file = prompts_dir / "ticket.diff"
    diff_file.write_text("diff --git a/main.py b/main.py")

    async def fake_invoke(request):
        assert request.models == ["groq/llama-3.3-70b-versatile"]
        assert request.options.temperature == 0.2
        assert request.options.max_tokens == 500
        assert request.options.retries == 3
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are precise about broken import."
        assert request.messages[1].role == "user"
        assert "Classify ticket 42." in request.messages[1].content
        assert "diff --git a/main.py b/main.py" in request.messages[1].content
        return InvokeResponse(
            model="groq/llama-3.3-70b-versatile",
            raw_text='{"tier":"B","reasoning":"uses a tool"}',
            structured={"tier": "B", "reasoning": "uses a tool"},
        )

    monkeypatch.setattr("llm_runner.run_templates.invoke_request", fake_invoke)

    response = asyncio.run(
        run_request(
            RunRequest(
                template={"path": str(run_template)},
                bindings={
                    "data": {"ticket": {"id": 42, "title": "broken import"}},
                    "text_files": [{"name": "diff", "path": str(diff_file)}],
                },
                overrides=RunOverrides(temperature=0.2),
            )
        )
    )

    assert response.run.template_path == str(run_template.resolve())
    assert response.run.model == "groq/llama-3.3-70b-versatile"
    assert response.response.structured == {"tier": "B", "reasoning": "uses a tool"}
    assert response.final_output.text is None
    assert response.final_output.data == {
        "ticket_id": 42,
        "tier": "B",
        "reasoning": "uses a tool",
    }
