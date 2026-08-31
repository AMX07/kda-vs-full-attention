#!/usr/bin/env bash
set -euo pipefail

for digits in 3 4 5; do
  .venv/bin/python -m karpathygpt_hw.variable_hybrid_addition \
    --max-digits "${digits}" \
    --output-dir "runs/variable-unpadded-reversed-hybrid/digits-${digits}" \
    --log-every 100
done
