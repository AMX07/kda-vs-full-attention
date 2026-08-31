# Full Attention vs Kimi-K3-Style Hybrid Attention on Addition

Investigation log. Date: August 24–26, 2026. Hardware: M5 Max (128 GB).

## Goal

Compare a dense causal Transformer against a Kimi-K3-style hybrid (3 KDA layers +
1 Gated MLA layer, NoPE) trained from scratch on decimal addition, looking for a
qualitative or quantitative difference in **task accuracy** and **compute
efficiency**.

Data protocol shared by every arm: character tokenizer, unpadded variable-length
rows `A+B=<reversed answer><EOS>`, answer-only masked loss, layout-balanced
sampling with a hard-carry fraction, held-out pair splits. Reversed answer digits
are the only inductive bias — no index hints, no column sinusoids, no operand
routing.

---

## Headline result

**At widths 6–9, full attention fails on mismatched-width layouts; the hybrid does not.**

Mean exact accuracy on the held-out test set, split by whether the operands have
equal digit counts (`runs/width-ladder/`, seed 4242, early stop at 3 consecutive
perfect quick evals, 100k cap):

| Width | FULL equal | FULL mismatched | gap | HYB equal | HYB mismatched | gap | FULL conv | HYB conv |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.0000 | 0.9985 | 0.002 | 0.9982 | 0.9983 | 0.000 | 10,250 | 11,750 |
| 6 | 0.9991 | 0.9029 | **0.096** | 1.0000 | 1.0000 | 0.000 | 16,500 | 20,500 |
| 7 | 0.9998 | 0.9173 | **0.083** | 1.0000 | 0.9994 | 0.001 | 21,250 | 38,000 |
| 8 | 0.9983 | 0.9701 | 0.028 | 1.0000 | 1.0000 | 0.000 | 24,000 | 14,250 |
| 9 | 1.0000 | 0.8181 | **0.182** | 1.0000 | 1.0000 | 0.000 | 43,750 | 27,750 |

At 9 digits the hybrid is perfect on all 171 layout groups (macro-layout 1.0).
Full attention's worst groups are all mismatched, e.g. `layout-6-5-6` at 3.1%
exact, `layout-5-6-6` at 4.7%.

**Proposed mechanism.** Rows are unpadded, so the absolute index of an arithmetic
column depends on operand widths — `123456+12345=` places `=` differently from
`123456+123456=`. Learned absolute position embeddings tie the operand-identification
circuit to absolute indices, so it must be relearned for each of the 78 (width 6)
to 171 (width 9) layouts. The hybrid is NoPE; its position sense comes from the
recurrence and is relative, so one "look back this far" pattern transfers across
every layout.

**Open caveat.** Both arms stopped on the same criterion — three consecutive
perfect *quick* evaluations — but `build_eval_groups(quick=True)` only tests
layouts where **both operands are at max width**. The hybrid happened to be
perfect on everything else when it triggered; full attention was not, and training
stopped regardless. So the established claim is *"full attention learns equal-width
layouts well before mismatched ones; the hybrid learns both together"* — a
sample-efficiency-per-layout difference — **not** "full attention cannot do
mismatched layouts." Fixing this requires an all-layout early-stop criterion and a
full-attention rerun.

**Confound to state in any writeup.** This compares *packages*: full attention
**with learned absolute positions** vs the hybrid **with NoPE**. The mechanism
implicates the position encoding, not the attention type. A NoPE or RoPE
full-attention arm would separate them and is the highest-value next experiment.

---

## Compute results

All measured at digits-5, 202k parameters, batch 256, identical data.

| Arm | Framework | ms/step | Peak memory |
|---|---|---|---|
| Full attention | MLX | **5.64** | **195 MiB** |
| Full attention | PyTorch / MPS | 15.44 | 297 MiB |
| Hybrid (KDA+MLA) | MLX | 9.50 | 407 MiB |

- **MLX is 2.74× faster than PyTorch/MPS** on an identical model, and uses 1.5×
  less memory. Running MLX with a manual matmul/mask/softmax attention gave 5.59
  ms/step — indistinguishable from the fused kernel — so none of the gap comes
  from attention kernel fusion.
  **Caveat: this is compiled MLX vs eager PyTorch.** Both MLX arms wrap the
  training step in `mx.compile`; the PyTorch arm uses no `torch.compile`. Since
  graph compilation is exactly what removes per-op dispatch overhead, an unknown
  share of the 2.74× is compilation rather than framework. Untested — a
  `torch.compile` baseline would be needed to separate them (its MPS support is
  limited). The **architecture** comparison is unaffected: both MLX arms call the
  identical compiled helper.
- **With framework controlled, full attention is ~1.75× cheaper per step than the
  hybrid at every width tested** (5.4–6.7 ms vs 9.4–12.8 ms), and uses 2.1× less
  memory. An earlier "hybrid is 1.7× faster" reading was purely PyTorch-vs-MLX
  overhead and **reversed sign** once corrected.

**Why linear attention loses here.** Per layer, attention costs ~`4·B·T²·d` FLOPs
while projections plus FFN cost ~`24·B·T·d²`; their ratio is `T / 6d`. This
experiment runs at T≈18–30 with d=64, so attention is ~5% of FLOPs and there is
essentially nothing for linear attention to save — meanwhile KDA pays for
depthwise convolutions, decay/write gates, and sequential state updates every
step. Linear attention only becomes profitable when T is several times d_model.

---

## Earlier phase: fixed-layout 1,364-digit runs (negative results)

Before the width ladder, three long-context attempts all failed. Documented in
`runs/runs/kda-mla-fixed-layout-1364/README.md` and
`runs/kda-mla-fixed-layout-halflife-1364/README.md`.

| Run | Outcome |
|---|---|
| Width curriculum (`kda-gated-mla-full`) | Solved 1→32 digits, died at 64, drifted to 128 at 0% exact |
| Fixed 1,364-digit layout, NoPE hybrid | Flat for 26,300 steps; loss pinned at 2.137 |
| Same + long decay half-life init (8→4096 tokens) | Flat for 6,250 steps; indistinguishable from baseline |

Every probe sat exactly on its **padding floor** (16-digit probe 0.989 vs floor
0.988; full-width 0.100 = chance). Two mechanisms were diagnosed as jointly
binding:

1. **Vanishing gradients through the KDA recurrence.** Default decay init (bias
   −4) gives per-step retention ≈0.914, a half-life of ~7.7 tokens; signal across
   the ~2,730-token operand→answer span is ~10⁻¹⁰⁷.
2. **Flat NoPE attention landscape.** The MLA layer's gradient w.r.t. *where* to
   attend carries no information: near-miss retrieval scores the same as random,
   and NoPE keys carry no positional features to address.

Fixing (1) alone via half-life init changed nothing, which is why (2) is the
live hypothesis. The curriculum and fixed-layout failures bracket the problem:
the curriculum had a bootstrap (short spans) but broke transfer at each width
promotion; the fixed layout preserved transfer but removed the bootstrap.

---

## Methodological traps hit (and how they were caught)

These cost the most time and are the most transferable lessons.

1. **Teacher-forced evaluation read as capability.** `evaluate()` scores a single
   forward pass on ground-truth inputs, so every "100%" gate never tested
   autoregressive generation. Teacher-forced accuracy upper-bounds strict
   accuracy. Training teacher-forced is correct and standard; using it as the
   success criterion is not.
2. **Padding-inflated accuracy.** Digit accuracy over a zero-padded answer slice
   has a floor set by the padding alone (0.988 at a 16-digit probe in a
   1,364-wide layout). Numbers "just above" that floor mean the model learned
   nothing real.
3. **Micro vs macro averaging.** `micro_answer_exact_accuracy` averages over all
   layout groups including trivial 1-digit ones. Reported 0.9254 for a model that
   was at 44.5% on the hard layouts. Use macro or per-group.
4. **Stopping mid-convergence.** The original 5,000-step comparison concluded
   "the hybrid can't do it"; at 40,000 steps the hybrid reaches 100% at every
   width 3–9. It was undertrained, not incapable.
5. **Cross-framework timing.** PyTorch-MPS vs MLX-Metal timings are not an
   architecture comparison. Any compute claim needs both arms in one framework.
6. **Silent init mismatch when porting.** PyTorch `nn.Embedding` defaults to
   N(0,1); MLX defaults to ≈1/√dims — an 8× difference that made the first MLX
   port look far slower to converge. Linear layers matched. Always compare init
   statistics after a port.
7. **GPU nondeterminism.** Neither framework is bitwise reproducible. Same seed,
   same config gave 0.66 and 1.00 at step 5,000 for the hybrid. Seeds do not
   fully control a run; single-run numbers near a transition are noise.
8. **Early-stop criterion that doesn't test the failure mode.** See the open
   caveat above — the quick eval omitted mismatched layouts, so full attention
   stopped while still failing them.

---

## Kimi K3 report, verified against the source

Checked against [arXiv 2607.24653](https://arxiv.org/pdf/2607.24653):

- **NoPE + KDA-supplied position is confirmed.** §2.1.2: *"The intervening KDA
  layers provide position-sensitive and recency-aware sequence mixing, while the
  MLA layers provide unrestricted global content interaction."* All MLA layers are
  NoPE.
- **Gated MLA is motivated by capacity, not gradient flow.** The report's stated
  reason for interleaving is "efficient long-context token mixing while retaining
  selective high-capacity attention," plus a final global layer. Vanishing
  gradients are never given as the motivation.
- **AttnRes addresses depth, not time.** It applies attention *across layers*, a
  bottleneck "reminiscent of RNNs over time." Irrelevant to our 4-layer models,
  whose failures are across the sequence.
- **The −5 decay bound is a numerics choice**, to keep cumulative decay within
  BF16 range so all chunk tiles use dense Tensor Core matmuls — not a memory-length
  choice.
- **The half-life-spread init we tried is NOT a K3 technique.** K3 initializes
  A_h=0 with biases following Kimi Linear/GDN convention. Log-spaced timescale
  init is imported from the SSM literature (S4/Mamba/LRU).

---

## Literature context

Closest published setups to ours (reversed format, standard PE, nothing else) are
Lee et al. 2023 (APE + reversed) and Kazemnejad et al. (NoPE + reversed).
[Zhou et al. 2024](https://arxiv.org/abs/2402.09371) ablate index hints and find
that without them, NoPE and FIRE show poor **in-distribution** accuracy at 40-digit
addition across all seeds, despite working at 10 digits — with a 25M-parameter
model. Methods that go further all add explicit operand-identification machinery:
position tokens (Nogueira et al., 60 digits), FIRE + randomized PE + index hints
(Zhou et al., train 40 → test 100), position coupling (train 30 → test 200),
Abacus embeddings (train 20 → test 100).

Note their setting trains one model over a mixed 1–40 range and tests
extrapolation; ours trains at fixed max width in-distribution, which is easier.
Our own earlier curriculum run reached 32 digits (teacher-forced).

---

## File index

**Models and training**

| Path | What |
|---|---|
| `src/karpathygpt_hw/kda_mlx.py` | KDA layer, fused Metal training kernel, decay parameterization |
| `src/karpathygpt_hw/hybrid_addition.py` | KDA+Gated-MLA model; fixed-layout and curriculum trainer |
| `src/karpathygpt_hw/long_addition.py` | Long-addition data generator, teacher-forced eval |
| `src/karpathygpt_hw/variable_addition.py` | Dense Transformer, PyTorch. Shared data/eval/layout code |
| `src/karpathygpt_hw/variable_hybrid_addition.py` | Hybrid arm, MLX |
| `src/karpathygpt_hw/variable_mlx_addition.py` | Dense Transformer, MLX. Parameter-identical port |
| `scripts/train_long_addition_full_attention_colab.py` | Standalone TPU/GPU full-attention script (paste into a Colab cell) |
| `scripts/run_width_ladder.sh` | The width ladder |
| `scripts/run_convergence_seed_sweep.sh` | Convergence + seed sweep (partially run) |

**Runs**

| Path | What |
|---|---|
| `runs/width-ladder/` | **Main result.** Widths 5–9, both arms, MLX |
| `runs/convergence-seed-sweep/` | 40k-step runs at widths 3–5; 7 complete |
| `runs/variable-unpadded-reversed{,-hybrid}/` | Original 5,000-step runs (superseded — undertrained) |
| `runs/runs/kda-mla-fixed-layout-1364/` | Flat-line control, has README |
| `runs/kda-mla-fixed-layout-halflife-1364/` | Half-life init, flat, has README |
| `runs/kda-gated-mla-full/` | Width curriculum, solved to 32 digits |

---

## Next steps, ranked

1. **All-layout early-stop criterion + full-attention rerun.** Determines whether
   the mismatched-layout gap closes with more training. Gates the headline claim.
2. **NoPE or RoPE full-attention arm.** Separates position encoding from attention
   type — currently the main confound.
3. **Seeds at the separating widths** (8–9). Everything above is n=1, and GPU
   nondeterminism is large.
4. **Strict autoregressive evaluation** on the final checkpoints, so no claim
   rests on teacher-forced numbers.
5. **Longer sequences before any compute-efficiency claim.** At T/6d ≈ 0.05 the
   hybrid cannot win; the argument needs T ≳ several × d_model, which means the
   H100 path (PyTorch + FLA's Triton `chunk_kda`, since MLX is Apple-only and
   FlashKDA is inference-only).
