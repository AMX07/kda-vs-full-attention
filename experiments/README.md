# Addition architecture experiments

This directory is the comparison registry for the long-addition work. Run
artifacts remain in `runs/`; this directory records which immutable checkpoint,
code, data representation, and metrics belong to each comparison baseline.

## Registered baselines

- `kda-aligned-context-1364/`: deployed four-layer KDA model with the
  addition-aligned operand-context feature.
- `kda-gated-mla-nope/`: shortcut-free three-KDA/one-Gated-MLA comparison,
  including its trial lineage and the still-open 1,364-digit target.

## Comparison rules

All architecture comparisons must report the following separately:

1. Parameter count and exact layer composition.
2. Training examples, optimizer updates, tokens, wall-clock time, and peak
   memory. Equal updates alone are not an apples-to-apples comparison when
   batch sizes differ.
3. Teacher-forced digit and exact accuracy.
4. Strict autoregressive exact accuracy on held-out random and structured
   cases.
5. Attention-core, full attention-layer, and complete-model latency as distinct
   scopes.
6. Whether the model was exposed to the evaluated digit width during training.
7. Every task-specific representation or architectural inductive bias.

The shared task representation is character-level tokenization and reversed
answer digits. Architecture experiments may not silently change either one.
