#!/usr/bin/env bash
set -uo pipefail

# Convergence + seed sweep for the full-attention vs NoPE-hybrid comparison.
#
# Fixes two problems with the 5,000-step single-seed runs:
#   1. The hybrid had not converged (digits-5 was still climbing at cutoff), so
#      the old result confounded "worse" with "slower".
#   2. n=1 made the hybrid's non-monotonic widths (4 digits beating 3) impossible
#      to attribute to variance.
#
# Both arms get the identical 40,000-step budget so neither is under-trained.

STEPS=40000
SEEDS=(4242 1337 2718)
DIGITS=(3 4 5)
ROOT="runs/convergence-seed-sweep"

for seed in "${SEEDS[@]}"; do
  for d in "${DIGITS[@]}"; do
    out="${ROOT}/full/digits-${d}-seed-${seed}"
    if [ -f "${out}/final_metrics.json" ]; then echo "skip ${out}"; continue; fi
    echo "=== FULL digits=${d} seed=${seed} ==="
    uv run python -m karpathygpt_hw.variable_addition \
      --max-digits "${d}" --steps "${STEPS}" --seed "${seed}" \
      --output-dir "${out}" --log-every 1000

    out="${ROOT}/hybrid/digits-${d}-seed-${seed}"
    if [ -f "${out}/final_metrics.json" ]; then echo "skip ${out}"; continue; fi
    echo "=== HYBRID digits=${d} seed=${seed} ==="
    .venv/bin/python -m karpathygpt_hw.variable_hybrid_addition \
      --max-digits "${d}" --steps "${STEPS}" --seed "${seed}" \
      --output-dir "${out}" --log-every 1000
  done
done
echo "SWEEP COMPLETE"
