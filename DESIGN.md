# llm-runner Design

## Purpose

`llm-runner` executes template-driven LLM tasks.

It depends on `llm-templating-engine` for all template concerns and owns:

- provider and model resolution
- API invocation
- retry and fallback policy
- structured output handling
- execution of template-defined runs
- optional response rendering through the templating engine

It does not own:

- prompt library management
- Jinja parsing or rendering rules
- template include/import semantics
- template variable materialization rules

Those belong in `llm-templating-engine`.

## Design Goals

- Make input templates the source of truth for runner behavior.
- Keep the canonical interop layer JSON-first for JS/TS callers.
- Keep the high-level API template-driven and the low-level API message-driven.
- Support structured model output without leaking Python-specific schema concepts into the
  public boundary.
- Allow post-processing through response templates without embedding templating logic in
  runner code.

## Runner Model

The runner has two public layers.

### High-Level Layer: Run a Template

The main product is a template-defined run.

The runner should:

- load an input template document through `llm-templating-engine`
- read runner metadata from that template's frontmatter
- materialize bindings through `llm-templating-engine`
- render prompt content
- invoke the selected model or fallback chain
- validate structured output if requested
- optionally render a response template through `llm-templating-engine`
- return a structured JSON result

### Low-Level Layer: Invoke a Model

The low-level API exists for callers that already have messages and do not need template
loading. It should be smaller and clearly secondary.

## Canonical Source of Runner Semantics

Runner semantics live in input template frontmatter.

That frontmatter should be treated as a run specification, not just passive metadata.

Suggested reserved fields for input templates:

```yaml
kind: llm-run
models:
  - openrouter/openai/gpt-4o-mini
  - groq/llama-3.3-70b-versatile
system_template:
  path: prompts/system/code-review.md
temperature: 0.0
max_tokens: 800
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
  path: prompts/responses/review-result.json.j2
  format: json
```

The body of the input template is the primary user prompt template.

## Public JSON Contracts

There should not be one generic `action` envelope. Each command should accept its own
typed JSON request.

## `run`

Execute a template-defined run.

Input:

```json
{
  "template": {
    "path": "prompts/runs/classify-ticket.md"
  },
  "bindings": {
    "data": {
      "ticket": {
        "id": 42,
        "title": "broken import"
      }
    },
    "text_files": [
      {
        "name": "diff",
        "path": "artifacts/current.diff"
      }
    ]
  },
  "overrides": {
    "models": ["groq/llama-3.3-70b-versatile"],
    "temperature": 0.1
  }
}
```

Output:

```json
{
  "run": {
    "template_path": "/abs/path/prompts/runs/classify-ticket.md",
    "model": "groq/llama-3.3-70b-versatile",
    "messages": [
      {
        "role": "system",
        "content": "You are precise."
      },
      {
        "role": "user",
        "content": "Classify ticket 42"
      }
    ]
  },
  "response": {
    "raw_text": "{\"tier\":\"B\",\"reasoning\":\"...\"}",
    "structured": {
      "tier": "B",
      "reasoning": "..."
    }
  },
  "final_output": {
    "text": null,
    "data": {
      "ticket_id": 42,
      "tier": "B",
      "reasoning": "..."
    }
  }
}
```

Interpretation:

- `response.raw_text` is the direct model output before any response templating.
- `response.structured` is the validated structured object when `output_schema` is set.
- `final_output` is the optional post-processed result from `response_template`.
  When it is structured, the payload lives under `final_output.data`.

## `invoke`

Call a model directly without loading a run template.

Input:

```json
{
  "models": [
    "openrouter/openai/gpt-4o-mini",
    "groq/llama-3.3-70b-versatile"
  ],
  "messages": [
    {
      "role": "system",
      "content": "You are precise."
    },
    {
      "role": "user",
      "content": "Classify this ticket."
    }
  ],
  "output_schema": {
    "type": "object",
    "properties": {
      "tier": { "type": "string" },
      "reasoning": { "type": "string" }
    },
    "required": ["tier", "reasoning"],
    "additionalProperties": false
  },
  "options": {
    "temperature": 0.0,
    "max_tokens": 500,
    "retries": 3
  }
}
```

Output:

```json
{
  "model": "groq/llama-3.3-70b-versatile",
  "raw_text": "{\"tier\":\"B\",\"reasoning\":\"...\"}",
  "structured": {
    "tier": "B",
    "reasoning": "..."
  }
}
```

This is the low-level escape hatch. It should stay smaller than `run`.

## `providers list`

Output should be JSON, not line-oriented text:

```json
{
  "providers": [
    {
      "name": "groq",
      "models": ["groq/llama-3.3-70b-versatile"]
    }
  ]
}
```

## Output Schema Design

The canonical public schema format should be JSON Schema.

That matters because:

- JS/TS callers already speak JSON Schema naturally
- template frontmatter can embed JSON Schema directly
- the public contract should not depend on Pydantic class names

Internal implementation may compile JSON Schema into Pydantic models or provider-specific
structured-output adapters, but those are internal details.

Named local schema registries may still exist as a convenience layer, but they should be
expressed as an explicit runner feature, not the primary public contract.

## Response Template Design

If `response_template` is present in input template frontmatter, the runner should render
that template after the model call using `llm-templating-engine`.

The response-template context should include:

- `request`: original run request
- `input`: parsed input template plus rendered prompt body
- `run`: resolved execution metadata such as chosen model
- `response.raw_text`
- `response.structured`

This keeps post-processing declarative and template-driven.

## CLI Shape

Primary CLI:

- `llm-runner run`
- `llm-runner invoke`
- `llm-runner providers list`

Standalone scripts:

- `llm-run`
- `llm-invoke`
- `llm-provider-list`

Each command should:

- read JSON from stdin by default
- write JSON to stdout by default
- optionally support `--input` and `--output`

There should be no generic `llm-runner-json` multiplexer command in the final design.

## Python Module Shape

Suggested module layout:

```text
src/llm_runner/
  __init__.py
  contracts.py
  schema_compiler.py
  providers.py
  invoke.py
  run_templates.py
  response_rendering.py
  cli.py
  cli_run.py
  cli_invoke.py
  cli_providers.py
```

This keeps the public code aligned with the actual product boundary:

- one module for low-level invocation
- one module for template-defined runs
- one module for schema compilation
- one module for response-template post-processing

## Separation of Concerns

`llm-runner` should consume `llm-templating-engine` like any other client.

It should call into the templating engine for:

- loading input templates
- rendering input prompt bodies
- rendering system templates
- rendering response templates

It should not duplicate:

- file-binding materialization
- template search-path logic
- Jinja environment setup
- frontmatter parsing

## Non-Goals

The runner should not try to be:

- a prompt catalog manager
- a workflow engine
- a chat session store
- a TS bridge layer with separate handwritten types

It is a Python runner with JSON contracts.

## Immediate Consequences for the Current Repo

The current ad hoc request envelope should be replaced.

Specifically:

- remove the single `JsonRequest` bag model
- remove the `action` switch multiplexer
- stop using the reserved `schema` field as a public request key
- move toward command-specific request and response models
- treat input template frontmatter as the canonical source of run configuration
