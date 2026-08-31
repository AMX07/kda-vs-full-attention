#!/usr/bin/env bash
set -euo pipefail

KGPT_RUN_ROOT="${1:-runs/kda-aligned-context-1364}"
KGPT_BOOTSTRAP_DIR="${KGPT_RUN_ROOT}/bootstrap"
KGPT_FINAL_DIR="${KGPT_RUN_ROOT}/final"
KGPT_BOOTSTRAP_CHECKPOINT="${KGPT_BOOTSTRAP_DIR}/checkpoint-00003000"
KGPT_FINAL_CHECKPOINT="${KGPT_FINAL_DIR}/checkpoint-00003200"

mkdir -p "${KGPT_BOOTSTRAP_DIR}" "${KGPT_FINAL_DIR}"

if [[ -f "${KGPT_FINAL_CHECKPOINT}/model.safetensors" ]]; then
  printf 'Training already complete: %s\n' "${KGPT_FINAL_CHECKPOINT}"
  exit 0
fi

if [[ ! -f "${KGPT_BOOTSTRAP_CHECKPOINT}/model.safetensors" ]]; then
  uv run --group apple kgpt-train-long-addition \
    --schedule 16:3000 \
    --position-encoding addition_column_context \
    --rehearsal-mode uniform_layout \
    --rehearsal-fraction 0.75 \
    --lr-decay-steps 250000 \
    --output-dir "${KGPT_BOOTSTRAP_DIR}" \
    --eval-every 500 \
    --checkpoint-every 1000 \
    2>&1 | tee "${KGPT_BOOTSTRAP_DIR}/training.jsonl"
fi

uv run --group apple kgpt-train-long-addition \
  --schedule 1364:200 \
  --position-encoding addition_column_context \
  --rehearsal-fraction 0 \
  --learning-rate 0.0003 \
  --lr-decay-steps 100000 \
  --carry-heavy-fraction 0.5 \
  --fork-from "${KGPT_BOOTSTRAP_CHECKPOINT}" \
  --output-dir "${KGPT_FINAL_DIR}" \
  --eval-batch-size 128 \
  --log-every 10 \
  --eval-every 100 \
  --checkpoint-every 100 \
  2>&1 | tee "${KGPT_FINAL_DIR}/training.jsonl"
