"""Typed public contracts for llm_runner."""

from __future__ import annotations

from typing import Any, Literal

from llm_templating_engine import Bindings, TemplateReference
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model for strict public request and response contracts."""

    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    """One chat message in the runner request/response model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class InvokeOptions(StrictModel):
    """Low-level model invocation options."""

    temperature: float = 0.0
    max_tokens: int = 500
    retries: int = 3


class InvokeRequest(StrictModel):
    """Request for direct model invocation."""

    models: list[str] = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    output_schema: dict[str, Any] | None = None
    options: InvokeOptions = Field(default_factory=InvokeOptions)


class InvokeResponse(StrictModel):
    """Response from direct model invocation."""

    model: str
    raw_text: str
    structured: Any | None = None


class RunOverrides(StrictModel):
    """Runtime overrides for a template-defined run."""

    models: list[str] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    retries: int | None = None
    output_schema: dict[str, Any] | None = None


class RunRequest(StrictModel):
    """Request for a template-defined run."""

    template: TemplateReference
    bindings: Bindings = Field(default_factory=Bindings)
    overrides: RunOverrides = Field(default_factory=RunOverrides)


class RunExecution(StrictModel):
    """Resolved run metadata."""

    template_path: str
    model: str
    messages: list[ChatMessage]


class FinalOutput(StrictModel):
    """Post-processed runner output."""

    text: str | None = None
    data: Any | None = None

    @model_validator(mode="after")
    def validate_output_shape(self) -> FinalOutput:
        """Require exactly one populated final output representation."""
        populated = int(self.text is not None) + int(self.data is not None)
        if populated != 1:
            raise ValueError("Exactly one of 'text' or 'data' must be populated.")
        return self


class RunResponse(StrictModel):
    """Response for a template-defined run."""

    run: RunExecution
    response: InvokeResponse
    final_output: FinalOutput


class ProviderInfo(StrictModel):
    """One provider entry in the provider catalog."""

    name: str
    models: list[str] = Field(default_factory=list)


class ProvidersListResponse(StrictModel):
    """Provider catalog response."""

    providers: list[ProviderInfo] = Field(default_factory=list)


class ErrorDetail(StrictModel):
    """Structured CLI error detail."""

    type: str
    message: str


class ErrorResponse(StrictModel):
    """Structured CLI error payload."""

    error: ErrorDetail


class ResponseTemplateSpec(StrictModel):
    """Frontmatter config for response-template rendering."""

    path: str | None = None
    text: str | None = None
    name: str | None = None
    format: Literal["text", "json"] = "text"

    @model_validator(mode="after")
    def validate_source(self) -> ResponseTemplateSpec:
        """Require exactly one template source."""
        source_count = int(self.path is not None) + int(self.text is not None)
        if source_count != 1:
            raise ValueError("Exactly one of 'path' or 'text' must be provided.")
        return self

    def as_template_reference(self) -> TemplateReference:
        """Convert to the templating engine's template reference model."""
        return TemplateReference(path=self.path, text=self.text, name=self.name)


class RunTemplateSpec(StrictModel):
    """Reserved runner frontmatter fields."""

    model_config = ConfigDict(extra="ignore")

    kind: str | None = None
    models: list[str] = Field(min_length=1)
    system_template: TemplateReference | None = None
    temperature: float = 0.0
    max_tokens: int = 500
    retries: int = 3
    output_schema: dict[str, Any] | None = None
    response_template: ResponseTemplateSpec | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> RunTemplateSpec:
        """Reject unsupported frontmatter kinds when kind is present."""
        if self.kind is not None and self.kind != "llm-run":
            raise ValueError(f"Unsupported template kind: {self.kind!r}")
        return self
