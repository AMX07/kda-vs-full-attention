# KDA Hybrid Attention vs Dense Softmax Attention

A controlled comparison of a **Kimi-K3-style hybrid** (3 Kimi Delta Attention layers +
1 Gated MLA layer, NoPE) against a **dense softmax Transformer** (learned absolute
positions), both trained from scratch on variable-length decimal addition.

Both arms are implemented in MLX and share one data pipeline, one compiled training
step, and one evaluation protocol, so the only difference is the model. The KDA layer
uses a custom fused Metal kernel with a hand-written VJP for the backward pass.

Full investigation log, including negative results and methodology notes:
**[docs/attention_comparison_log.md](docs/attention_comparison_log.md)**

---

## Headline result

**The hybrid was more accurate; the dense Transformer was cheaper.** That is the
inverse of the usual expectation for linear attention.

Exact-match accuracy, macro-averaged over all layout groups on a held-out test split
(~200–240k parameters per arm, identical data, same early-stop criterion):

| Operand width | Dense softmax | KDA hybrid |
|---|---|---|
| 5 digits | 0.9988 | 0.9981 |
| 6 digits | 0.9188 | **1.0000** |
| 7 digits | 0.9290 | **0.9995** |
| 8 digits | 0.9736 | **1.0000** |
| 9 digits | **0.8383** | **1.0000** |

At 9 digits the hybrid was perfect on all 171 layout groups.

## Why: absolute positions do not transfer across layouts

Splitting the test set by whether the two operands have the *same* digit count
isolates the failure:

| Width | Softmax, equal-width | Softmax, mismatched | Gap | Hybrid gap |
|---|---|---|---|---|
| 6 | 0.9991 | 0.9029 | 0.096 | 0.000 |
| 7 | 0.9998 | 0.9173 | 0.083 | 0.001 |
| 8 | 0.9983 | 0.9701 | 0.028 | 0.000 |
| 9 | 1.0000 | 0.8181 | **0.182** | 0.000 |

The dense model is flawless when operands are the same length and collapses when they
differ — `layout-6-5-6` scored 3.1% exact. The hybrid shows no gap at any width.

Rows are unpadded, so the absolute index of a given arithmetic column shifts with
operand width: `123456+12345=` places `=` differently from `123456+123456=`. Learned
absolute position embeddings tie the operand-identification circuit to absolute
indices, so it must be relearned for each of the 78 (width 6) to 171 (width 9)
layouts. The hybrid is NoPE — its position sense comes from the recurrence and is
relative, so a single "look back this far" pattern transfers across every layout.

## Compute: the hybrid lost on every axis at this scale

| Metric (digits-5, MLX, matched data) | Dense softmax | KDA hybrid |
|---|---|---|
| Time per step | **5.6 ms** | 9.5 ms |
| Peak memory | **195 MiB** | 407 MiB |
| Parameters | 202,125 | 238,921 |

Dense attention stayed ~1.75× cheaper per step at every width tested (5.4–6.7 ms vs
9.4–12.8 ms).

This is expected, not a defect of the kernel. Per layer, attention costs roughly
`4·B·T²·d` FLOPs while projections plus the feed-forward cost `24·B·T·d²`; the ratio
is `T / 6d`. These experiments run at T≈18–30 with d=64, so attention is about **5% of
FLOPs** — there is essentially nothing for linear attention to save, while KDA still
pays for depthwise convolutions, decay and write gates, and sequential state updates.
**Linear attention only becomes profitable when T is several times d_model.**

---

## Caveats

These bound what the results above support.

1. **The early-stop criterion did not test the failure mode.** Both arms stopped after
   three consecutive perfect *quick* evaluations, but that evaluation only covers
   layouts where both operands are at max width. The hybrid happened to be perfect
   everywhere else when it triggered; the dense model was not, and training stopped
   anyway. So the supported claim is *"the dense model learns equal-width layouts well
   before mismatched ones"* — a sample-efficiency difference — **not** that it cannot
   learn them given more steps.
2. **This compares packages, not attention mechanisms.** Dense softmax **with learned
   absolute positions** vs hybrid **with NoPE**. The mechanism implicates the position
   encoding, not the attention type. A NoPE or RoPE dense arm would separate them and
   is the most valuable next experiment.
3. **n=1 per cell.** Neither MLX/Metal nor PyTorch/MPS is bitwise reproducible; the
   same seed and config produced 0.66 and 1.00 at step 5,000 in one case. Single-run
   numbers near a convergence transition are noise.
4. **Evaluation is teacher-forced** except where noted. Teacher-forced accuracy upper-
   bounds strict autoregressive accuracy.
5. **Short sequences only.** No claim here transfers to the long-context regime where
   linear attention is designed to win.

---

## Reproducing

Requires Apple silicon (MLX + Metal).

```bash
uv sync --group apple
```

Run the full width ladder (both arms, widths 5–9, ~1.5–2 h):

```bash
./scripts/run_width_ladder.sh
```

Or a single arm at one width:

```bash
# dense softmax
uv run --group apple python -m karpathygpt_hw.variable_mlx_addition \
  --max-digits 9 --steps 100000 --seed 4242 --output-dir runs/demo-dense

# KDA + Gated MLA hybrid
uv run --group apple python -m karpathygpt_hw.variable_hybrid_addition \
  --max-digits 9 --steps 100000 --seed 4242 --output-dir runs/demo-hybrid
```

Each run writes `metrics.jsonl` (per-step training and per-eval validation),
`final_metrics.json` (per-layout test breakdown), and model weights.

```bash
uv run pytest
```

## Task and data format

Character-level, unpadded, variable-length rows:

```
A+B=<reversed answer><EOS>
```

Answers are reversed so carries propagate left-to-right, matching autoregressive
generation order. This is the **only** inductive bias — no index hints, no
arithmetic-column position encodings, no operand-to-answer routing. Loss is masked to
the answer digits and the terminal token. Batches are layout-balanced with a
configurable hard-carry fraction, and train/test pairs are split by a hash of the
operand pair.

## Layout

| Path | Contents |
|---|---|
| `src/karpathygpt_hw/kda_mlx.py` | KDA layer, fused Metal training kernel, custom VJP |
| `src/karpathygpt_hw/hybrid_addition.py` | 3×KDA + 1×Gated-MLA model and long-addition trainer |
| `src/karpathygpt_hw/variable_mlx_addition.py` | Dense softmax arm (MLX) |
| `src/karpathygpt_hw/variable_hybrid_addition.py` | Hybrid arm (MLX) |
| `src/karpathygpt_hw/variable_addition.py` | Dense arm (PyTorch) + shared data, layouts, evaluation |
| `scripts/run_width_ladder.sh` | The main experiment |
| `docs/attention_comparison_log.md` | Full log: results, negative results, methodology |
| `runs/` | Metrics and per-run notes (model weights are gitignored) |

Note: the Python package is still named `karpathygpt_hw` — this project began as a
Karpathy GPT homework exercise before becoming an architecture comparison.

## Background

The hybrid follows the attention-layer pattern described in the Kimi K3 technical
report ([arXiv:2607.24653](https://arxiv.org/pdf/2607.24653)): three KDA layers per
Gated MLA layer, with all MLA layers using NoPE on the premise that *"the intervening
KDA layers provide position-sensitive and recency-aware sequence mixing."* It does not
reproduce K3's MoE, Attention Residuals, or multimodal components.

On length generalization in arithmetic Transformers, see
[Zhou et al. 2024](https://arxiv.org/abs/2402.09371),
[Lee et al. 2023](https://arxiv.org/abs/2307.03381), and
[McLeish et al. 2024](https://arxiv.org/abs/2405.17399).
