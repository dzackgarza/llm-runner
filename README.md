# LLM Runner

Template-driven LLM execution built on top of `llm-templating-engine`.

## Setup

```bash
direnv allow
just setup
```

Local configuration lives in `.envrc`:

```bash
export PROMPTS_DIR="${PROMPTS_DIR:-$PWD/prompts}"
# export GROQ_API_KEY="replace-me"
# export OPENROUTER_API_KEY="replace-me"
```

## Direct Use

```bash
uvx --from git+https://github.com/dzackgarza/llm-runner.git llm-run --help

uvx --from git+https://github.com/dzackgarza/llm-runner.git llm-invoke --help

uvx --from git+https://github.com/dzackgarza/llm-runner.git llm-provider-list --help
```

## Commands

- `llm-run` executes a template-defined run from JSON on stdin.
- `llm-invoke` calls models directly from JSON on stdin.
- `llm-provider-list` emits provider and model data as JSON.

## Development

- `just setup` installs the project and dev dependencies.
- `just check` runs lint and tests.
- `just build` builds a publication-ready wheel and sdist.
- `just bump` increments the minor version with `uv version --bump minor`.

Full interface and contract details live in `DESIGN.md`.
