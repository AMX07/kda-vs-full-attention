.PHONY: sync sync-apple env-check lab test lint format strip-notebooks

sync:
	uv sync

sync-apple:
	uv sync --group apple

env-check:
	uv run python scripts/check_env.py

lab:
	uv run jupyter lab

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

strip-notebooks:
	uv run nbstripout notebooks/*.ipynb
