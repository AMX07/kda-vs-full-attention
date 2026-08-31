#!/usr/bin/env bash
set -euo pipefail

# Controlled counterpart to the reversed-answer runs: only answer order changes.

uv run python -m karpathygpt_hw.variable_addition \
  --answer-order normal \
  --max-digits 3 \
  --output-dir runs/variable-unpadded-normal/digits-3 \
  --log-every 100

uv run python -m karpathygpt_hw.variable_addition \
  --answer-order normal \
  --max-digits 4 \
  --output-dir runs/variable-unpadded-normal/digits-4 \
  --log-every 100

uv run python -m karpathygpt_hw.variable_addition \
  --answer-order normal \
  --max-digits 5 \
  --output-dir runs/variable-unpadded-normal/digits-5 \
  --log-every 100
