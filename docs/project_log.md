# Project Setup Log

Date: May 23, 2026

## Repository Structure

Created a clean learning-in-public layout:

- `homework/` for the four Karpathy GPT exercises.
- `notebooks/` for exploratory and Colab-style work.
- `src/karpathygpt_hw/` for reusable Python code.
- `scripts/` for runnable utilities.
- `data/`, `models/`, and `runs/` for local artifacts ignored by git.
- `docs/` for setup, compute, workflow, dependencies, and this log.
- `tests/` for lightweight verification.

## Environment

Installed `uv` using Astral's official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Created the project environment:

```bash
uv sync
uv sync --group apple
```

`uv` downloaded CPython 3.12.13 and created `.venv`.

## Dependencies

Added `pyproject.toml` and `uv.lock`.

Installed the main stack:

- PyTorch, NumPy, pandas, pyarrow, matplotlib, scikit-learn.
- tiktoken, tokenizers, sentencepiece.
- datasets, transformers, accelerate, safetensors.
- TensorBoard, Weights & Biases, rich, tqdm, Typer.
- JupyterLab, notebook, ipykernel, ipywidgets.
- Jedi language server for richer notebook Python help.
- pytest, ruff, mypy, pre-commit, nbstripout.

Installed the optional Apple Silicon stack:

- `mlx`
- `mlx-metal`

## Helper Code

Added `src/karpathygpt_hw/device.py`, which chooses the best available PyTorch device in this order:

```text
cuda -> mps -> cpu
```

Added `scripts/check_env.py` and a console script named `kgpt-check-env`.

## Verification

Environment check:

```text
python: 3.12.13
platform: macOS-26.4-arm64-arm-64bit
machine: arm64
torch: 2.12.0
device: mps
cuda_available: False
mps_available: True
mps_built: True
```

Checks run:

```bash
uv run python scripts/check_env.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Result: all passed.
