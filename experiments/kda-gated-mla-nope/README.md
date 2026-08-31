# Shortcut-free KDA + Gated MLA comparison

This registry entry records the separate hybrid architecture and every trial
used to develop its training recipe. It does not replace or modify the deployed
aligned-context checkpoint.

## Locked architecture and task representation

- Four layers: KDA, KDA, KDA, then global causal Gated MLA.
- 64 hidden dimensions, four heads, 238,921 parameters.
- NoPE: no RoPE and no task-specific positional signal.
- No arithmetic-column sinusoid.
- No aligned-operand projection or other direct operand-to-answer routing.
- The same 13-token character-index tokenizer as the baseline.
- The same `A+B=reverse(sum)` representation, answer-only next-token loss,
  carry-heavy generator, optimizer family, and validation definitions.
- The KDA blocks use the fused Metal forward/custom-VJP training path. The MLA
  block uses MLX causal scaled dot-product attention with low-rank Q and shared
  low-rank KV sources plus a learned output gate.

## Trial lineage and measured results

1. `kda-gated-mla-smoke-current` verified the final training, validation,
   adaptive-batch, checkpoint, best-pointer, and durable-log paths over 20
   updates. The earlier `kda-gated-mla-smoke-logged` run is retained as history.
2. `kda-gated-mla-1-16-trial` trained widths 3, 8, and 16 for 10,000 updates.
   The mixed fixed-width/rehearsal recipe did not solve 16 digits.
3. `kda-gated-mla-focus-8` forked the 3-digit checkpoint and solved most held-out
   8-digit batches; its best recorded validation was 96.875% random exact and
   100% carry-chain exact at global step 10,000.
4. `kda-gated-mla-focus-16` forked step 10,000 and reached 100% teacher-forced
   exact accuracy on both 64-example random and carry-chain batches at steps
   20,000 through 28,000. The selected step-28,000 checkpoint precedes a
   high-learning-rate collapse at step 28,500.
5. A deterministic length sweep of the selected checkpoint reproduced 100%
   exact accuracy at 16 digits but 0% at 17, 32, 64, 128, and 256 digits. This
   is evidence of fixed-layout learning, not a length-independent algorithm.
6. `kda-gated-mla-1364-trial` forked step 28,000 and ran 500 target-width
   updates at batch size 4. It measured about 159 ms/update and 103k tokens/s,
   but ended at 0% exact accuracy (10.37% random digit accuracy and 28.48%
   carry-chain digit accuracy). This was a runnable scaling trial, not a
   successful 1,364-digit checkpoint.

The current single training script therefore has a validated differentiable
architecture and a full target-width curriculum, but matching the deployed
baseline's 1,364-digit accuracy has not yet been demonstrated. No result in
this directory should be described as such until a full run passes the same
teacher-forced and strict autoregressive checks recorded for the baseline.

## Full comparison run

```bash
uv run --group apple kgpt-train-long-addition-hybrid \
  --preset full \
  --eval-batch-size 64 \
  --output-dir runs/kda-gated-mla-full
```

The default batch strategy uses 64 examples through 128 digits, then tapers to
32, 16, 8, and 4 at widths 256, 512, 1,024, and 1,364. This preserves efficient
short-width learning while bounding Gated MLA's quadratic attention memory.
Every stage records the actual batch size. Two consecutive perfect random and
carry validations advance a solved stage; pass `--early-stop-exact-evals 0` to
disable this behavior.

The trial hashes, deterministic sweep, feature flags, and source hashes are in
`manifest.json` and `length-sweep.json`.
