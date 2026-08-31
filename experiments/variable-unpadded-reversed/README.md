# Variable-length, unpadded, reversed addition

This experiment isolates the effect of removing the original notebook's fixed-width
zero-padding while retaining its reversed-answer objective.

## Controlled architecture

- character-level decoder-only Transformer
- dense causal self-attention
- learned absolute position embeddings
- width 64, 4 heads, 4 layers
- four-times-wide ReLU feed-forward layers
- pre-norm residual blocks and zero dropout
- AdamW, learning rate `1e-3`, weight decay `0.01`
- answer-only next-token cross-entropy

These choices match `notebooks/gpt-hw.ipynb` except that the batch size is increased
from 16 to 256 for better M5 Max utilization.

## Data representation

Rows have the form `A+B=reverse(A+B)<space>`. There are no leading zeroes and no
answer-width zero-padding. For example:

```text
7+23=03 
12+3=51 
123+456=975 
```

Operand-length pairs and feasible answer lengths are balanced. Twenty-five percent
of training batches contain a carry through every active column. Train, validation,
and test pairs are separated by a deterministic hash.

Every batch is drawn from a single exact `(digits A, digits B, answer digits)` layout.
Sequence lengths therefore vary across batches, but no tensor contains padding tokens.

## Runs

Execute all three fresh-model runs with:

```bash
bash scripts/run_variable_unpadded_reversed_3_to_5.sh
```

Each run writes `config.json`, `metrics.jsonl`, `model.pt`, and `final_metrics.json`
under `runs/variable-unpadded-reversed/digits-N/`.

The individual commands used on 2026-08-26 were:

```bash
.venv/bin/python -m karpathygpt_hw.variable_addition \
  --max-digits 3 \
  --output-dir runs/variable-unpadded-reversed/digits-3 \
  --log-every 100

.venv/bin/python -m karpathygpt_hw.variable_addition \
  --max-digits 4 \
  --output-dir runs/variable-unpadded-reversed/digits-4 \
  --log-every 100

.venv/bin/python -m karpathygpt_hw.variable_addition \
  --max-digits 5 \
  --output-dir runs/variable-unpadded-reversed/digits-5 \
  --log-every 100
```

All runs were fresh starts with seed 4242, batch size 256, and 5,000 optimizer
steps. They ran sequentially on the M5 Max through PyTorch 2.12's MPS backend.

Before the measured runs, a 100-step 3-digit smoke test exercised training,
checkpointing, and autoregressive evaluation end to end. It took 2.24 seconds to
train and, as expected for a preflight rather than a convergence run, achieved
1.46% exact accuracy on 480 test cases. Its artifacts are retained under
`runs/variable-unpadded-reversed/smoke/` but it is excluded from the result table.

## Results

Accuracy below is autoregressive answer accuracy: the model receives only `A+B=`
and must greedily generate the complete reversed answer plus the terminating space.
The test pairs are disjoint from the train and validation pairs.

| Maximum operand digits | Layouts | Maximum model input length | Parameters | Held-out test cases | Exact answers | Per-digit accuracy | Full-carry exact | Training time | Final evaluation | Total time | Peak MPS driver memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 18 | 12 | 201,741 | 3,223 | 100.000% | 100.000% | 100.000% | 70.82 s | 1.41 s | 72.23 s | 173.2 MiB |
| 4 | 32 | 15 | 201,933 | 6,559 | 100.000% | 100.000% | 100.000% | 76.66 s | 1.43 s | 78.09 s | 223.0 MiB |
| 5 | 50 | 18 | 202,125 | 10,922 | 99.918% | 99.980% | 100.000% | 78.23 s | 1.89 s | 80.12 s | 298.9 MiB |

The quick validation suite first recorded 100% exact-answer accuracy at steps
2,250, 4,250, and 4,750 for the 3-, 4-, and 5-digit runs respectively. This is
not an early-stopping claim: all models completed the fixed 5,000-step protocol.

At five digits, 10,913 of 10,922 held-out answers were exact. The nine failures
were distributed over six ordinary layouts; all five full-carry stress groups were
perfect. The 3- and 4-digit models were exact on every final test case.

## What changed from the original notebook

The original 3-digit notebook always uses the 12-token shifted sequence derived
from `DDD+DDD=DDDD `, including leading operand zeroes and a four-character answer.
This experiment instead includes all operand widths from one through the selected
maximum. For the 3-digit run, for example, the model sees real forms such as `7+23=`
rather than converting that prompt to `007+023=`. Removing the padding expands the
3-digit task from one aligned layout to 18 exact length layouts.

The model architecture, answer reversal, answer-only loss, and AdamW optimizer are
preserved. Batch size is the one deliberate throughput change: 256 here versus 16
in the notebook. Each batch contains one layout, so that increase does not reintroduce
padding.

## Interpretation and limitations

- These results show that zero-padded fixed alignment is not required at 3--5 digits
  when reversed answers, balanced layouts, and explicit carry examples are retained.
- Reversal is still a strong alignment aid: output token 1 corresponds to the units
  column even though the `=` position varies. This experiment does not test normal-
  order answers.
- Twenty-five percent of training batches deliberately emphasize complete carry
  chains. The experiment therefore tests unpadded length variation, not whether a
  completely uniform sampler discovers rare carries unaided.
- Validation and test examples are generated deterministically, not a stored corpus.
  Small finite layouts may have fewer than the requested 256 distinct test pairs;
  this is why test-suite sizes differ by run.
- One seed was run for each setting. The timing and convergence points are measurements,
  but seed-to-seed variance has not yet been estimated.
- MPS execution is asynchronous; the runner synchronizes before timing boundaries and
  reports the largest driver allocation observed at its sampling points. These are
  local M5 Max measurements, not portable hardware guarantees.
- This is an interpolation test up to each model's trained maximum length. It does not
  establish extrapolation to longer, unseen operand widths.
