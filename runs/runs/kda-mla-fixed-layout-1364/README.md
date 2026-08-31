# Fixed-layout NoPE hybrid, default init — flat-line control

**Date:** 2026-08-24 · **Status:** terminated at step 26,300 of 100,000
(user decision; zero trend). Last checkpoint: `checkpoint-00026000`.

```
uv run --group apple kgpt-train-long-addition-hybrid \
  --preset fixed --batch-size 32 --eval-batch-size 16 \
  --output-dir runs/kda-mla-fixed-layout-1364    # launched from runs/, hence nested path
```

NoPE 3×KDA + 1×Gated-MLA (238,921 params) on fixed-layout 1,364-digit
addition: every example zero-padded to T=4,095, operand width sampled per
step (25% full width, else uniform), 50% carry-heavy, answer-only loss.
Motivation: the width-curriculum run (`kda-gated-mla-full`) solved 1–32
digits but died at every promotion; freezing the operand→answer geometry
removes that non-stationarity.

**Result: zero learning in 26,300 steps (~840k examples).** Full-width train
loss flat at median 2.137 (uniform chance ln 10 ≈ 2.303; the gap is marginal
structure — padding zeros, carry-chain patterns). Every teacher-forced probe
(16/64/512/1364 active digits) pinned to its padding floor from the first
evaluation; full-width digit accuracy = 0.10 = chance. Exact accuracy 0.000
everywhere.

**Interpretation:** stationary geometry alone does not enable the NoPE stack
to bootstrap a positional code precise enough for ~2,730-token retrieval.
Paired with the curriculum run, this brackets the failure: the curriculum
had a bootstrap (short spans) but broke transfer; fixed layout preserves
transfer but removes the bootstrap. See
`../../kda-mla-fixed-layout-halflife-1364/README.md` for the follow-up
(long decay half-life init — also flat) and the next planned arms.
