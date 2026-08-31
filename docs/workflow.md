# Workflow

This repo is meant to be both a homework archive and a public learning trail. Favor small, readable commits that show the thinking.

## Daily Loop

1. Create or update the relevant homework README.
2. Explore in `notebooks/` if the idea is still fluid.
3. Move reusable pieces into `src/karpathygpt_hw/`.
4. Put one-off runnable commands in `scripts/`.
5. Save large artifacts under `data/`, `models/`, or `runs/`.
6. Commit code, configs, and notes.

## Learning-in-Public Template

Each exercise README should eventually answer:

- What was I trying to learn?
- What implementation did I try?
- What command did I run?
- What changed in the loss, samples, or speed?
- What confused me?
- What would I try next?

## Git Hygiene

Commit these:

- Source code.
- Small configs.
- Notes and write-ups.
- Tiny sample text outputs.
- Dependency files: `pyproject.toml`, `uv.lock`, `.python-version`.

Do not commit these:

- Large datasets.
- Checkpoints.
- TensorBoard logs.
- `.venv`.
- API keys or secrets.

## Notebook Hygiene

Notebooks are welcome, especially for Colab. Before committing large notebooks:

```bash
uv run nbstripout notebooks/*.ipynb
```

When the same code is used twice, promote it into `src/karpathygpt_hw/`.
