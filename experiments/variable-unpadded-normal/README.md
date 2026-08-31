# Variable-length, unpadded, normal-order addition

This is the controlled counterpart to `experiments/variable-unpadded-reversed/`.
It changes only the answer order: the model generates the conventional most-
significant digit first rather than reversing the sum.

Rows have forms such as:

```text
7+23=30 
12+3=15 
123+456=579 
```

There are no leading operand zeroes, answer-width padding, or padding tokens in a
batch. All model, optimizer, sampling, data-split, seed, batch-size, and step-count
settings are identical to the reversed-answer runs.

Run all three fresh models with:

```bash
bash scripts/run_variable_unpadded_normal_3_to_5.sh
```

The exact commands used on 2026-08-26 were:

```bash
.venv/bin/python -m karpathygpt_hw.variable_addition \
  --answer-order normal --max-digits 3 \
  --output-dir runs/variable-unpadded-normal/digits-3 --log-every 100

.venv/bin/python -m karpathygpt_hw.variable_addition \
  --answer-order normal --max-digits 4 \
  --output-dir runs/variable-unpadded-normal/digits-4 --log-every 100

.venv/bin/python -m karpathygpt_hw.variable_addition \
  --answer-order normal --max-digits 5 \
  --output-dir runs/variable-unpadded-normal/digits-5 --log-every 100
```

Each was an independent fresh start with seed 4242, batch size 256, and 5,000
AdamW steps. The runs executed sequentially on the M5 Max through PyTorch 2.12 MPS.

## Results

Evaluation is autoregressive and greedy: the model receives only `A+B=` and must
generate the entire answer plus its terminating space. All reported examples are
from the held-out hash split.

| Maximum operand digits | Parameters | Test cases (all groups) | Exact, all groups | Per-digit, all groups | Exact, ordinary layouts | Exact, max-width `D+D` | Full-carry exact (macro) | Training time | Total time | Max observed driver memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 201,741 | 3,223 | 91.592% | 96.891% | 90.255% | 81.055% | 88.759% | 70.56 s | 71.27 s | 173.2 MiB |
| 4 | 201,933 | 6,559 | 56.289% | 83.693% | 52.899% | 16.406% | 84.094% | 75.47 s | 76.67 s | 223.0 MiB |
| 5 | 202,125 | 10,922 | 34.875% | 71.758% | 29.305% | 4.297% | 92.908% | 80.56 s | 82.25 s | 298.9 MiB |

"All groups" includes the ordinary layout groups and separately sampled hard-carry
groups. The ordinary-layout column removes those stress groups. The max-width column
is the most revealing scale measure: it includes only held-out examples where both
operands have the run's maximum width, split across answers with `D` and `D+1` digits.

For example, the 5-digit model is not a 34.9%-accurate five-digit adder. That aggregate
also includes the easier one- through four-digit layouts. On actual `5-digit +
5-digit` cases it produced 22 of 512 complete answers correctly (4.297%).

## Direct controlled comparison with reversed answers

| Maximum digits | Normal-order exact, all groups | Reversed exact, all groups | Accuracy-point gap | Normal max-width exact |
|---:|---:|---:|---:|---:|
| 3 | 91.592% | 100.000% | -8.408 | 81.055% |
| 4 | 56.289% | 100.000% | -43.711 | 16.406% |
| 5 | 34.875% | 99.918% | -65.043 | 4.297% |

Training latency is essentially unchanged by output order. The large difference is
learning efficiency, not kernel cost: reversing the answer makes autoregressive
generation follow the natural units-to-most-significant carry direction.

## Learning behavior

The best quick-validation exact accuracy occurred at step 4,750 in every normal-order
run: 92.19%, 48.70%, and 36.72% for maximum widths 3, 4, and 5. Final quick-validation
accuracy at step 5,000 was lower: 88.02%, 43.75%, and 30.47%. The constant `1e-3`
learning rate inherited from the original notebook produces visible late-run
volatility; these runs intentionally retain the final checkpoint instead of selecting
a validation winner.

## Interpretation and limitations

- The models are learning: digit accuracy remains well above chance and the 3-digit
  model exceeds 90% exact accuracy across ordinary layouts. Normal order is not
  impossible, but it is dramatically less sample-efficient under this architecture.
- In conventional addition the first generated digit can depend on carries originating
  at the far-right end of the operands. Reversed output exposes result digits in the
  same direction as the carry recurrence; normal output asks the fixed-depth network
  to resolve those dependencies before emitting its first token.
- The strong hard-carry scores should not be mistaken for general accuracy. A quarter
  of training batches explicitly contain full carry chains, and those examples have a
  distinctive structure. Ordinary arbitrary max-width additions remain much harder.
- Only one seed and one fixed 5,000-step budget were tested. A learning-rate decay,
  validation checkpoint selection, larger model, longer run, or scratchpad/carry
  tokens could improve normal-order results, but each would be a new experiment rather
  than this controlled output-order comparison.
- These are interpolation evaluations through the trained maximum width; no claim is
  made about extrapolation to unseen lengths.
