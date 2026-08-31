# Setup

This project uses `uv` for Python, virtual environments, dependency locking, and local commands.

## 1. Make uv Available

`uv` was installed with Astral's official installer into:

```text
/Users/anshmittal/.local/bin
```

In a new terminal, run:

```bash
source "$HOME/.local/bin/env"
```

Add that line to your shell profile later if you want it available in every terminal automatically.

## 2. Create the Environment

From the repo root:

```bash
uv sync --group apple
```

That reads `pyproject.toml`, creates `.venv`, resolves dependencies into `uv.lock`, and installs the default plus dev dependency groups. The `apple` group adds MLX for Apple Silicon experiments.

On a Linux GPU server, use the portable default:

```bash
uv sync
```

## 3. Verify Local Acceleration

```bash
uv run python scripts/check_env.py
```

On an Apple Silicon Mac, the preferred local training device should be `mps` when PyTorch's Metal backend is available. On an NVIDIA remote box, it should be `cuda`.

## 4. Jupyter

```bash
uv run jupyter lab
```

For Colab, keep notebooks in `notebooks/`, then move reusable code into `src/karpathygpt_hw/` so local, Colab, and remote machines all run the same implementation.

## 5. What Was Installed Locally

This machine now has:

- `uv` and `uvx` in `/Users/anshmittal/.local/bin`.
- A repo-local `.venv` managed by uv.
- CPython 3.12.13 in the uv environment.
- PyTorch with MPS available.
- Jupyter, Hugging Face tooling, experiment tracking, linting, testing, and formatting tools.
- Optional Apple Silicon MLX packages via `uv sync --group apple`.

## References

- uv project dependencies: https://docs.astral.sh/uv/concepts/projects/dependencies/
- uv virtual environments: https://docs.astral.sh/uv/pip/environments/
- PyTorch MPS notes: https://docs.pytorch.org/docs/stable/notes/mps.html
