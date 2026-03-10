"""Response-template rendering helpers for llm_runner."""

from __future__ import annotations

import json
from typing import Any

from llm_templating_engine import Bindings, RenderTemplateRequest, render_template

from llm_runner.contracts import FinalOutput


def render_response_template(template, *, context: dict[str, Any]) -> FinalOutput:
    """Render a response template into text or JSON final output."""
    rendered = render_template(
        RenderTemplateRequest(
            template=template.as_template_reference(),
            bindings=Bindings(data=context),
        )
    ).rendered.body
    if template.format == "json":
        return FinalOutput(text=None, data=json.loads(rendered))
    return FinalOutput(text=rendered, data=None)


__all__ = ["render_response_template"]
