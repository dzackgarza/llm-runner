"""Template-defined runs for llm_runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_templating_engine import (
    Bindings,
    RenderTemplateRequest,
    TemplateDocument,
    TemplateReference,
    render_template,
)

from llm_runner.contracts import (
    ChatMessage,
    FinalOutput,
    InvokeOptions,
    InvokeRequest,
    RunExecution,
    RunRequest,
    RunResponse,
    RunTemplateSpec,
)
from llm_runner.invoke import invoke_request
from llm_runner.response_rendering import render_response_template


def _template_base_directory(document: TemplateDocument) -> Path | None:
    """Return the directory used to resolve related templates."""
    identifier = document.path or document.name
    if identifier is None:
        return None
    return Path(identifier).expanduser().resolve().parent


def _resolve_related_template(
    reference: TemplateReference,
    document: TemplateDocument,
) -> TemplateReference:
    """Resolve a relative frontmatter template path against the run template."""
    if reference.path is None:
        return reference
    candidate = Path(reference.path).expanduser()
    if candidate.is_absolute():
        return reference
    base_directory = _template_base_directory(document)
    if base_directory is None:
        return reference
    return TemplateReference(path=str((base_directory / candidate).resolve()))


def _apply_overrides(spec: RunTemplateSpec, request: RunRequest) -> RunTemplateSpec:
    """Merge runtime overrides onto the template frontmatter spec."""
    override_payload: dict[str, Any] = {}
    if request.overrides.models is not None:
        override_payload["models"] = request.overrides.models
    if request.overrides.temperature is not None:
        override_payload["temperature"] = request.overrides.temperature
    if request.overrides.max_tokens is not None:
        override_payload["max_tokens"] = request.overrides.max_tokens
    if request.overrides.retries is not None:
        override_payload["retries"] = request.overrides.retries
    if request.overrides.output_schema is not None:
        override_payload["output_schema"] = request.overrides.output_schema
    return spec.model_copy(update=override_payload)


def _system_message(
    spec: RunTemplateSpec,
    document: TemplateDocument,
    bindings: Bindings,
) -> ChatMessage | None:
    """Render the optional system template into a system chat message."""
    if spec.system_template is None:
        return None
    rendered = render_template(
        RenderTemplateRequest(
            template=_resolve_related_template(spec.system_template, document),
            bindings=bindings,
        )
    )
    return ChatMessage(role="system", content=rendered.rendered.body)


def _default_final_output(response) -> FinalOutput:
    """Return the default final output when no response template is configured."""
    if response.structured is not None:
        return FinalOutput(text=None, data=response.structured)
    return FinalOutput(text=response.raw_text, data=None)


async def run_request(request: RunRequest) -> RunResponse:
    """Execute a template-defined runner request."""
    rendered_input = render_template(
        RenderTemplateRequest(
            template=request.template,
            bindings=request.bindings,
        )
    )
    document = rendered_input.template
    spec = _apply_overrides(
        RunTemplateSpec.model_validate(document.frontmatter),
        request,
    )

    messages: list[ChatMessage] = []
    system_message = _system_message(spec, document, request.bindings)
    if system_message is not None:
        messages.append(system_message)
    messages.append(ChatMessage(role="user", content=rendered_input.rendered.body))

    response = await invoke_request(
        InvokeRequest(
            models=spec.models,
            messages=messages,
            output_schema=spec.output_schema,
            options=InvokeOptions(
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                retries=spec.retries,
            ),
        )
    )

    final_output = _default_final_output(response)
    if spec.response_template is not None:
        final_output = render_response_template(
            template=spec.response_template.model_copy(
                update={
                    "path": (
                        str(
                            (
                                _template_base_directory(document)
                                / spec.response_template.path
                            ).resolve()
                        )
                        if spec.response_template.path is not None
                        and not Path(spec.response_template.path).is_absolute()
                        and _template_base_directory(document) is not None
                        else spec.response_template.path
                    )
                }
            ),
            context={
                "request": request,
                "input": {
                    "template": rendered_input.template,
                    "prompt_body": rendered_input.rendered.body,
                    "prompt_document": rendered_input.rendered.document,
                },
                "run": RunExecution(
                    template_path=document.path or document.name or "",
                    model=response.model,
                    messages=messages,
                ),
                "response": response,
            },
        )

    return RunResponse(
        run=RunExecution(
            template_path=document.path or document.name or "",
            model=response.model,
            messages=messages,
        ),
        response=response,
        final_output=final_output,
    )


__all__ = ["run_request"]
