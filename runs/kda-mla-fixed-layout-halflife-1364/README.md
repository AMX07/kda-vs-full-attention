# Fixed-layout NoPE hybrid + long decay half-life init — negative result

**Date:** 2026-08-24 · **Hardware:** M5 Max 128GB, MLX fused KDA Metal kernel
**Status:** terminated at step ~6,250 of 100,000 (flat; see interpretation)

## Command

```
uv run --group apple kgpt-train-long-addition-hybrid \
  --preset fixed --batch-size 32 --eval-batch-size 16 \
  --decay-half-life-min 8 --decay-half-life-max 4096 \
  --output-dir runs/kda-mla-fixed-layout-halflife-1364
```

## Hypothesis

The control run (`runs/runs/kda-mla-fixed-layout-1364`, terminated flat at step
26,300) trains the NoPE 3×KDA + 1×Gated-MLA hybrid (238,921 params, d=64,
4 heads) on fixed-layout 1,364-digit addition (T=4,095, zero-padded operands,
random per-step widths) and never leaves the initialization basin. Diagnosis
identified two candidate mechanisms:

1. **Vanishing gradients through the KDA recurrence.** Default decay init
   (bias −4) gives per-step retention ≈ 0.914 (half-life ≈ 7.7 tokens);
   signal/gradient across the ~2,730-token operand→answer span is ~10⁻¹⁰⁷.
2. **Flat NoPE attention landscape.** The MLA layer's gradient w.r.t. *where*
   to attend is uninformative: near-miss retrieval scores the same as random,
   and keys carry no positional features to address.

This run tests mechanism 1's fix in isolation: identical everything, except
KDA decay biases log-spaced so channel half-lives span 8→4,096 tokens
(slowest channels retain 63% of signal across the full 2,730-token span at
init, vs ~0 in the control).

## Result

Indistinguishable from the control through 6,250 steps (~200k examples):

| Metric | Control (26k steps) | This run (6k steps) |
|---|---|---|
| Full-width train loss (median) | 2.137, flat | 2.139, flat |
| Digit acc, 16-digit probe (padding floor 0.988) | 0.989 | 0.989 |
| Digit acc, 64-digit probe (floor 0.952) | 0.958 | 0.957 |
| Digit acc, 512-digit probe (floor 0.624) | 0.662 | 0.662 |
| Digit acc, full width (chance 0.10) | 0.100 | 0.100 |
| Exact accuracy, any probe | 0.000 | 0.000 |

All validation is teacher-forced (upper bound on strict autoregressive).
~583 ms/step at batch 32, T=4,095. Terminated early: with zero loss movement
to the fourth decimal and the control as prior, further steps had negligible
evidence value.

## Interpretation

Long-memory initialization alone does not restore gradient traction. The slow
channels *can* transport information across the operand→answer span, but
nothing teaches the network what to write into them or how to read them out —
consistent with mechanisms 1 and 2 being jointly binding rather than
alternatives. The live hypothesis is now mechanism 2: without positional
signal that attention can exploit smoothly (RoPE/learned positions), no
gradient path leads from marginal-statistics loss (~2.14, vs ln 10 ≈ 2.30
uniform chance) to a retrieval circuit.

Note: this half-life-range init is *not* a Kimi K3 technique — K3 initializes
A_h=0 with biases following Kimi Linear/GDN convention, and its −5 decay
bound exists for BF16/Tensor-Core numerics, not memory length. The flag is a
homegrown import of standard SSM timescale-spread init (S4/Mamba/LRU style).

## Next arms (in order of diagnostic value)

1. Full attention + learned absolute positions, same fixed-layout data —
   `scripts/train_long_addition_full_attention_colab.py` (Colab v5e-1 TPU).
   Tests whether position-equipped global attention learns the task at all.
2. RoPE on the MLA layer only, NoPE KDA preserved — minimal concession
   targeting mechanism 2 directly.
3. Operand-digit auxiliary loss — dense supervision to bootstrap the
   retrieval circuit while keeping the architecture NoPE.
