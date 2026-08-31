#!/usr/bin/env bash
set -euo pipefail

# Three independent runs.  Keeping them independent avoids transferring learned
# weights from shorter to longer layouts and makes the results directly comparable.

uv run python -m karpathygpt_hw.variable_addition \
  --max-digits 3 \
  --output-dir runs/variable-unpadded-reversed/digits-3 \
  --log-every 100

uv run python -m karpathygpt_hw.variable_addition \
  --max-digits 4 \
  --output-dir runs/variable-unpadded-reversed/digits-4 \
  --log-every 100

uv run python -m karpathygpt_hw.variable_addition \
  --max-digits 5 \
  --output-dir runs/variable-unpadded-reversed/digits-5 \
  --log-every 100
