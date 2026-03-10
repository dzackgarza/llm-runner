default:
    @just --list

setup:
    uv sync --dev

test:
    uv run pytest

lint:
    uv run ruff check src tests

format:
    uv run ruff format src tests

check: lint test

build:
    uv build --no-sources

bump:
    uv version --bump minor
    uv lock

clean:
    python -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in [Path('dist'), Path('build'), Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache')]]; [shutil.rmtree(path, ignore_errors=True) for path in Path('.').glob('*.egg-info')]"
