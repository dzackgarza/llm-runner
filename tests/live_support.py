"""Shared live-test helpers for llm_runner."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import textwrap

from llm_runner.providers import list_models

SYSTEM_PROMPT = "Repeat the requested data exactly."
DIFF_TEXT = "diff --git a/main.py b/main.py"


def exact_echo_schema() -> dict[str, object]:
    """Return the structured-output schema used in live tests."""
    return {
        "type": "object",
        "properties": {
            "passphrase": {"type": "string"},
            "number": {"type": "integer"},
        },
        "required": ["passphrase", "number"],
        "additionalProperties": False,
    }


def exact_echo_user_prompt(passphrase: str, number: int) -> str:
    """Build a deterministic structured-output request."""
    return (
        "Return JSON with exact fields passphrase and number. "
        f"Use passphrase {passphrase} and number {number}."
    )


@lru_cache(maxsize=1)
def live_model_slug() -> str:
    """Pick one real Groq model from the live provider catalog."""
    available = list_models("groq")
    if not available:
        raise AssertionError("No live Groq models available in the current environment.")
    preferred = (
        "groq/llama-3.3-70b-versatile",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/openai/gpt-oss-120b",
    )
    for slug in preferred:
        if slug in available:
            return slug
    return available[0]


def write_exact_echo_templates(tmp_path: Path, *, model_slug: str) -> tuple[Path, Path]:
    """Write a live run-template bundle and return template and diff paths."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "system.md").write_text(SYSTEM_PROMPT)
    (prompts_dir / "response.json.j2").write_text(
        textwrap.dedent(
            f"""
            {{
              "passphrase": {{{{ response.structured.passphrase | tojson }}}},
              "number": {{{{ response.structured.number }}}},
              "diff_present": {{{{ ({DIFF_TEXT!r} in input.prompt_body) | tojson }}}}
            }}
            """
        ).strip()
    )
    run_template = prompts_dir / "echo.md"
    run_template.write_text(
        textwrap.dedent(
            f"""
            ---
            name: exact_echo
            description: Live runner fixture with extra non-runner frontmatter
            inputs:
              - name: passphrase
                required: true
              - name: number
                required: true
            kind: llm-run
            models:
              - {model_slug}
            system_template:
              path: system.md
            output_schema:
              type: object
              properties:
                passphrase:
                  type: string
                number:
                  type: integer
              required: [passphrase, number]
              additionalProperties: false
            response_template:
              path: response.json.j2
              format: json
            ---
            Return JSON with exact fields passphrase and number.
            Use passphrase {{{{ passphrase }}}} and number {{{{ number }}}}.
            {{{{ diff }}}}
            """
        ).strip()
    )
    diff_file = prompts_dir / "ticket.diff"
    diff_file.write_text(DIFF_TEXT)
    return run_template, diff_file
