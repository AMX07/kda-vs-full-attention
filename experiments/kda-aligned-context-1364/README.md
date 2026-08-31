# KDA aligned-context 1,364-digit baseline

This is the immutable baseline used by the local addition interface. The files
listed in `manifest.json` remain in their original locations; SHA-256 hashes
detect accidental changes without duplicating large checkpoints.

## Result and scope

- Model: four KDA blocks, 64 hidden dimensions, four heads, NoPE backbone.
- Task-specific feature: learned projections route the aligned A/B operand
  embeddings to answer-query positions.
- Parameters: 237,917.
- Deployed checkpoint: global step 3,200. It was forked from an aligned-context
  step-3,000 checkpoint and therefore contains 200 saved updates at 1,364
  digits (12,800 examples at batch size 64).
- Training input length: 4,095 tokens. The complete serialized row including
  the terminal space is 4,096 characters.
- Recorded step-3,200 validation: 100% teacher-forced exact accuracy on 128
  random examples and 128 full carry-chain examples.
- Post-training strict checks: 22/22 fixed-width 1,364-digit random/structured
  cases and 22/22 width-sweep cases were exact. The reproducibility seeds and
  case definitions are recorded in `manifest.json`.

This baseline was trained at the target width and must not be described as
zero-shot 1,364-digit extrapolation.

