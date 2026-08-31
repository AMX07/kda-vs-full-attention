# Dependencies

The environment is intentionally broad enough for the GPT homework without turning into a giant research stack.

## Core ML

- `torch`: lecture implementation, local MPS, and remote CUDA.
- `numpy`, `pandas`, `pyarrow`: data handling.
- `matplotlib`, `scikit-learn`: analysis and plots.
- `tqdm`, `rich`: progress and readable terminal output.

## Language Modeling

- `tiktoken`: GPT-style tokenization.
- `tokenizers`, `sentencepiece`: alternate tokenizer experiments.
- `datasets`: Hugging Face datasets.
- `transformers`, `accelerate`, `safetensors`: comparison with modern transformer tooling and easier remote scaling.
- `einops`: readable tensor rearrangements for attention work.

## Experiment Tracking

- `tensorboard`: local metrics.
- `wandb`: optional hosted experiment tracking.
- `python-dotenv`: local `.env` config.

## Dev Tools

- `jupyterlab`, `notebook`, `ipykernel`, `ipywidgets`: notebooks and Colab-style iteration.
- `jedi-language-server`: richer Python completion, hover, and signature help in JupyterLab/Notebook 7.
- `pytest`: tests.
- `ruff`: linting and formatting.
- `mypy`: type checking when useful.
- `nbstripout`: clean notebook outputs.
- `pre-commit`: optional local checks before commits.

## Optional Apple Stack

The `apple` dependency group includes `mlx` for Apple Silicon experiments. The main homework path stays PyTorch-first because the lecture, nanoGPT, Colab, and remote GPUs all align naturally around PyTorch.

Install the optional Apple group with:

```bash
uv sync --group apple
```
