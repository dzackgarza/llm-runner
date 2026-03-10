# LLM Runner

Template-driven LLM execution built on top of `llm-templating-engine`.

## Setup

```bash
direnv allow
just setup
```

Local configuration lives in `.envrc` and inherits shared API keys from `~/.envrc`:

```bash
source_up
export PROMPTS_DIR="${PROMPTS_DIR:-$PWD/prompts}"

# Shared provider credentials come from ~/.envrc:
# GROQ_API_KEY
# OPENROUTER_API_KEY
# NVIDIA_NIM_API_KEY
# MISTRAL_API_KEY
# CLOUDFLARE_API_KEY
# CLOUDFLARE_ACCOUNT_ID
# OLLAMA_API_KEY
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
