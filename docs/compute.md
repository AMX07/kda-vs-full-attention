# Compute Strategy

The goal is to use the Mac aggressively for fast iteration while keeping the repo ready for Colab or a remote GPU when a run outgrows local memory or patience.

## Local Mac

Use local compute for:

- Tensor shape drills and EX1.
- Small GPTs on Tiny Shakespeare.
- Dataset exploration.
- Debugging training loops.
- Short sweeps that finish in minutes to a few hours.

PyTorch should use the `mps` backend on Apple Silicon:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
```

The helper in `src/karpathygpt_hw/device.py` chooses `cuda`, then `mps`, then `cpu`.

## Colab

Use Colab when:

- You want a quick hosted GPU without provisioning a server.
- You are sharing a notebook publicly.
- You are trying a mid-sized run but do not need long-running reliability.

Workflow:

1. Pull the repo into Colab or upload a notebook from `notebooks/`.
2. Install from the project metadata when practical.
3. Save durable outputs back to GitHub, Google Drive, or another storage location.

## Remote GPU

Use a remote GPU instance when:

- The local Mac is memory-limited.
- A run needs many hours.
- You need NVIDIA CUDA performance for larger GPT-2-style experiments.

Suggested remote bootstrap:

```bash
git clone <your-repo-url>
cd KarpathyGPT-HW
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync
uv run python scripts/check_env.py
```

Keep datasets and checkpoints outside git. Commit configs, scripts, notes, and small sample outputs.

## Practical Rule

Prototype locally, scale remotely, then write down what happened. The committed repo should explain the experiment even when the heavy artifacts live elsewhere.
