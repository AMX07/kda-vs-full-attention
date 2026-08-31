# Addition Transformer

Karpathy GPT homework notebook for learning decoder-only Transformers by building a small character-level model that learns addition.

The main notebook is [notebooks/gpt-hw.ipynb](notebooks/gpt-hw.ipynb).

## What This Covers

- **EX1: Parallel multi-head attention**
  - Starts from the lecture-style separate `Head` and `MultiHeadAttention` classes.
  - Merges the attention computation so all heads are computed together with tensor operations.
  - Practices the core tensor shape flow: `(B, T, C) -> (B, n_head, T, head_size)`.

- **EX2: Addition as language modeling**
  - Generates synthetic addition examples up to 3-digit numbers.
  - Hard-codes a tiny character vocabulary: digits, `+`, `=`, and space.
  - Trains a decoder-only Transformer to predict reversed answer digits, e.g. `123+456=9750 ` represents answer `0579`.

## Model Sketch

The model is a compact GPT-style decoder:

```text
token ids
-> token embeddings + learned position embeddings
-> repeated Transformer blocks
   -> pre-norm causal multi-head self-attention
   -> residual connection
   -> pre-norm feed-forward network
   -> residual connection
-> final layer norm
-> vocab logits
```

Current notebook configuration:

```text
vocab_size = 13
n_embd = 64
n_head = 4
n_layer = 4
parameters ~= 0.2M
```

## Result

The saved notebook run reaches roughly:

```text
train loss ~= 0.0003
val loss   ~= 0.0003
```

The inference cells then prompt the model with examples like:

```text
123+456=
```

and autoregressively generate the answer digits.

## Long-addition KDA training

The MLX implementation also includes a fused Metal forward/backward KDA path
and an on-the-fly curriculum trainer. Start with the safe 40-step smoke run:

```bash
uv run --group apple kgpt-train-long-addition
```

The first serious curriculum progresses from 3 to 1,364-digit operands while
mixing random examples, full carry chains, and padded rehearsal examples:

```bash
uv run --group apple kgpt-train-long-addition \
  --preset full \
  --output-dir runs/kda-long-addition-full
```

This preset contains 60,000 optimizer steps. A steady-state stage-by-stage
measurement on the M5 Max totals about 3.0 hours of training compute; allow
roughly 3–3.5 hours after validation, checkpointing, and stage compilation.

Override the schedule for focused experiments:

```bash
uv run --group apple kgpt-train-long-addition \
  --schedule 3:500,8:1000,16:2000 \
  --batch-size 64
```

Checkpoints contain the model, AdamW state, curriculum position, and random
generator state. Resume from a checkpoint directory with `--resume PATH`.

### Validated 1,364-digit recipe

The original NoPE all-KDA model develops position-specific addressing gaps as
the operands grow. The opt-in `addition_column_context` mode gives each answer
query direct, linear-time access to its matching A/B digit pair while leaving
carry propagation to the fused KDA recurrence. It adds no quadratic attention
matrix and keeps the original reference model as the default.

Run the validated two-phase recipe with:

```bash
scripts/train_kda_1364.sh
```

It first learns the local addition/carry rule on mixed 1–16-digit layouts, then
fine-tunes for 200 updates at 1,364 digits (model-input length 4,095). On the M5
Max, the long phase measured about 0.53 seconds per optimizer step and completed
in roughly two minutes including validation/checkpointing. The saved run reached
100% teacher-forced exact accuracy on two independent validation rounds totaling
256 random additions and 256 forced carry-chain additions at 1,364 digits.
Strict autoregressive checks and artifact hashes are recorded under
`experiments/kda-aligned-context-1364/`. This mode is task-specific: the aligned
routing uses the known `A+B=reverse(sum)` layout and is not a general
language-model feature.

### NoPE KDA + Gated MLA comparison

The separate hybrid trainer follows the compact K3 layer pattern—three fused
KDA blocks followed by one global Gated MLA block—and deliberately exposes no
arithmetic-column sinusoid or aligned-operand routing option. It keeps the same
13-character tokenizer, reversed-answer target, loss, optimizer, and validation
definitions as the KDA baseline.

Run a smoke test with:

```bash
uv run --group apple kgpt-train-long-addition-hybrid \
  --preset smoke \
  --output-dir runs/kda-gated-mla-smoke
```

Start the complete target-width curriculum with:

```bash
uv run --group apple kgpt-train-long-addition-hybrid \
  --preset full \
  --output-dir runs/kda-gated-mla-full
```

The default hybrid has 238,921 parameters versus 237,917 for the deployed KDA
baseline. Its Gated MLA attention width is expanded while its Q and KV sources
use 32-dimensional latents. The default adaptive strategy uses batch size 64
through 128 digits and tapers to 4 at 1,364 digits because the MLA layer has
quadratic attention compute. Every run writes its actual per-stage batch size,
configuration, architecture feature flags, metrics, best-checkpoint pointers,
and checkpoint lineage to `training.jsonl` and checkpoint `state.json` files.

The shortcut-free architecture has reached 100% teacher-forced exact accuracy
at 16 digits in a focused trial. It did not extrapolate to 17 digits, and a
500-update 1,364-digit scaling trial remained at 0% exact accuracy. The full
curriculum is the recipe to test target-width capability; equivalence to the
deployed aligned-context checkpoint is not yet a measured result. The complete
lineage and hashes are recorded under `experiments/kda-gated-mla-nope/`.

Launch the local model-and-verifier interface with:

```bash
uv run --group apple kgpt-addition-ui --open
```

It accepts full decimal operands or bounded expressions such as
`(10^1364 - 1) + (10^1364 - 1)`. Cached recurrent decoding produces the model
answer in about 1.5 seconds at maximum length on the M5 Max, after which Python
exact arithmetic independently verifies every output digit.

## Notes

This is intentionally an exploratory learning notebook, not a polished training library. The goal is to make the Transformer mechanics visible: causal masking, attention heads, tensor reshaping, residual connections, loss masking, and autoregressive generation.
