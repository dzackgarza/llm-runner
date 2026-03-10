"""CLI tests for the JSON command surfaces."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from llm_runner.cli import app, invoke_app, run_app
from tests.live_support import (
    DIFF_TEXT,
    SYSTEM_PROMPT,
    exact_echo_schema,
    exact_echo_user_prompt,
    live_model_slug,
    write_exact_echo_templates,
)


runner = CliRunner()


def test_umbrella_help_lists_new_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "invoke" in result.stdout
    assert "providers" in result.stdout


def test_run_command_reads_json_from_stdin(tmp_path) -> None:
    model_slug = live_model_slug()
    passphrase = "PASS_CLI_RUN_20260310"
    number = 17
    template_path, diff_path = write_exact_echo_templates(
        tmp_path, model_slug=model_slug
    )
    result = runner.invoke(
        run_app,
        [],
        input=json.dumps(
            {
                "template": {"path": str(template_path)},
                "bindings": {
                    "data": {"passphrase": passphrase, "number": number},
                    "text_files": [{"name": "diff", "path": str(diff_path)}],
                },
            }
        ),
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["run"]["model"] == model_slug
    assert output["run"]["messages"][0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    assert DIFF_TEXT in output["run"]["messages"][1]["content"]
    assert output["response"]["structured"] == {
        "passphrase": passphrase,
        "number": number,
    }
    assert output["final_output"]["data"] == {
        "passphrase": passphrase,
        "number": number,
        "diff_present": True,
    }


def test_invoke_command_reads_json_from_stdin() -> None:
    model_slug = live_model_slug()
    passphrase = "PASS_CLI_INVOKE_20260310"
    number = 19
    result = runner.invoke(
        invoke_app,
        [],
        input=json.dumps(
            {
                "models": [model_slug],
                "messages": [
                    {"role": "system", "content": "Return only the requested JSON fields."},
                    {
                        "role": "user",
                        "content": exact_echo_user_prompt(passphrase, number),
                    },
                ],
                "output_schema": exact_echo_schema(),
                "options": {"temperature": 0.0, "max_tokens": 80, "retries": 2},
            }
        ),
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["model"] == model_slug
    assert output["structured"] == {"passphrase": passphrase, "number": number}
    assert json.loads(output["raw_text"]) == output["structured"]


def test_providers_list_command_emits_json() -> None:
    model_slug = live_model_slug()
    result = runner.invoke(app, ["providers", "list"])

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    providers = {provider["name"]: provider["models"] for provider in output["providers"]}
    assert model_slug in providers["groq"]
