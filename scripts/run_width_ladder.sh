#!/usr/bin/env bash
set -uo pipefail

# Width ladder: full attention vs K3-style NoPE hybrid, both in MLX.
#
# Purpose is a SCALING CURVE, not a single accuracy number. Both arms train at
# each width until the quick evaluation holds perfect for three consecutive
# checks (or the cap is hit), and the recorded convergence step is the primary
# comparison metric. Widths 3-5 are already known to saturate for both arms, so
# the ladder starts at 5 as an anchor and climbs into the separation zone.
#
# Both arms are MLX so wall-clock and memory are also comparable.

CAP=100000
SEED=4242
ROOT="runs/width-ladder"

for d in 5 6 7 8 9; do
  out="${ROOT}/full/digits-${d}"
  if [ ! -f "${out}/final_metrics.json" ]; then
    echo "=== FULL(MLX) digits=${d} ==="
    uv run python -m karpathygpt_hw.variable_mlx_addition \
      --max-digits "${d}" --steps "${CAP}" --seed "${SEED}" \
      --output-dir "${out}" --log-every 5000
  fi

  out="${ROOT}/hybrid/digits-${d}"
  if [ ! -f "${out}/final_metrics.json" ]; then
    echo "=== HYBRID digits=${d} ==="
    .venv/bin/python -m karpathygpt_hw.variable_hybrid_addition \
      --max-digits "${d}" --steps "${CAP}" --seed "${SEED}" \
      --output-dir "${out}" --log-every 5000
  fi
done
echo "LADDER COMPLETE"
