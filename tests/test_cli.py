"""CLI tests for the JSON command surfaces."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from llm_runner.cli import app, invoke_app, run_app


runner = CliRunner()


def test_umbrella_help_lists_new_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "invoke" in result.stdout
    assert "providers" in result.stdout


def test_run_command_reads_json_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    response_payload = {
        "run": {
            "template_path": "/tmp/prompts/classify-ticket.md",
            "model": "groq/llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": "Classify ticket 42."},
            ],
        },
        "response": {
            "model": "groq/llama-3.3-70b-versatile",
            "raw_text": '{"tier":"B","reasoning":"uses a tool"}',
            "structured": {"tier": "B", "reasoning": "uses a tool"},
        },
        "final_output": {
            "text": None,
            "data": {"tier": "B"},
        },
    }

    async def fake_execute(request):
        assert request.template.path == "prompts/classify-ticket.md"
        from llm_runner.contracts import RunResponse

        return RunResponse.model_validate(response_payload)

    monkeypatch.setattr("llm_runner.cli.execute_run_request", fake_execute)

    result = runner.invoke(
        run_app,
        [],
        input=json.dumps(
            {
                "template": {"path": "prompts/classify-ticket.md"},
                "bindings": {"data": {"ticket": {"id": 42}}},
            }
        ),
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["run"]["model"] == "groq/llama-3.3-70b-versatile"
    assert output["final_output"]["data"] == {"tier": "B"}


def test_invoke_command_reads_json_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_execute(request):
        assert request.models == ["groq/llama-3.3-70b-versatile"]
        from llm_runner.contracts import InvokeResponse

        return InvokeResponse(
            model="groq/llama-3.3-70b-versatile",
            raw_text='{"tier":"B","reasoning":"uses a tool"}',
            structured={"tier": "B", "reasoning": "uses a tool"},
        )

    monkeypatch.setattr("llm_runner.cli.execute_invoke_request", fake_execute)

    result = runner.invoke(
        invoke_app,
        [],
        input=json.dumps(
            {
                "models": ["groq/llama-3.3-70b-versatile"],
                "messages": [{"role": "user", "content": "Classify the ticket."}],
                "output_schema": {
                    "type": "object",
                    "properties": {"tier": {"type": "string"}},
                    "required": ["tier"],
                    "additionalProperties": False,
                },
            }
        ),
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["model"] == "groq/llama-3.3-70b-versatile"
    assert output["structured"] == {"tier": "B", "reasoning": "uses a tool"}


def test_providers_list_command_emits_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list_providers():
        from llm_runner.contracts import ProviderInfo, ProvidersListResponse

        return ProvidersListResponse(
            providers=[
                ProviderInfo(
                    name="groq",
                    models=["groq/llama-3.3-70b-versatile"],
                )
            ]
        )

    monkeypatch.setattr("llm_runner.cli.list_providers_response", fake_list_providers)

    result = runner.invoke(app, ["providers", "list"])

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output == {
        "providers": [
            {
                "name": "groq",
                "models": ["groq/llama-3.3-70b-versatile"],
            }
        ]
    }
