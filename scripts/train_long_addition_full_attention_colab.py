"""Full-attention long-addition training — paste into a single Colab cell.

A scale-up of the 3-digit addition transformer (learned absolute position
embeddings, pre-norm causal attention) to fixed-layout 1,364-digit addition.
The data pipeline is an exact port of the local hybrid experiments — reversed
zero-padded answers, per-step random operand widths, carry-heavy batches,
answer-only masked loss, teacher-forced probes at 16/64/512/1364 digits —
so results are directly comparable against the KDA/Gated-MLA runs.

Colab usage (Runtime -> Change runtime type -> v5e-1 TPU):
  1. (optional, recommended) In a first cell, mount Drive so checkpoints
     survive disconnects:
         from google.colab import drive; drive.mount('/content/drive')
  2. Paste this whole file into a cell, edit CONFIG below, run.
  3. After a disconnect, set CONFIG["resume"] to the latest checkpoint path
     and run the cell again.
Auto-detects TPU (torch_xla) -> CUDA -> CPU; the same cell works on a T4.
Parameter parity with the 238k hybrid: dims=64, heads=4, layers=4.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

CONFIG = {
    "digits": 1364,
    "steps": 100_000,
    # TPU note: PyTorch SDPA has no flash kernel on XLA, so each layer stores a
    # full B x heads x T x T attention matrix for backward (~270MB per batch row
    # at T=4095 across 6 layers). batch 4 x accum 8 keeps the effective batch
    # at 32 while peaking around 9-10GB of the v5e-1's 16GB HBM.
    "batch_size": 4,
    "grad_accum": 8,
    "dims": 128,
    "heads": 8,
    "layers": 6,
    "dropout": 0.0,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
    "full_width_fraction": 0.25,
    "carry_heavy_fraction": 0.5,
    "eval_batch_size": 8,
    "log_every": 50,
    "eval_every": 500,
    "strict_eval_every": 5000,
    "checkpoint_every": 1000,
    "output_dir": "/content/drive/MyDrive/full-attn-1364",  # Drive path recommended
    "resume": None,  # e.g. "/content/drive/MyDrive/full-attn-1364/checkpoint-00012000.pt"
    "seed": 4242,
}

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    XLA = True

    def xla_sync():
        # torch_xla >= 2.4 prefers torch_xla.sync(); older builds use mark_step.
        (torch_xla.sync if hasattr(torch_xla, "sync") else xm.mark_step)()

except ImportError:
    XLA = False

    def xla_sync():
        pass

TOKENS = (" ", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=")
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
DIGIT = TOKEN_TO_ID["0"]
IGNORE = -1


def sequence_length(digits: int) -> int:
    return 3 * digits + 3  # aaa+bbb=cccc, shifted target adds the final space


# ---------------------------------------------------------------- data (port)
def _carry_chain_operands(rng, rows, digits):
    a = np.zeros((rows, digits), dtype=np.int64)
    b = np.zeros((rows, digits), dtype=np.int64)
    for position in range(digits - 1, -1, -1):
        if position == digits - 1:
            a_col = rng.integers(1, 10, size=rows)
            min_b = 10 - a_col
        else:
            a_col = rng.integers(0, 10, size=rows)
            min_b = np.maximum(0, 9 - a_col)
        b_col = min_b + (rng.random(rows) * (10 - min_b)).astype(np.int64)
        a[:, position] = a_col
        b[:, position] = b_col
    return a, b


def make_addition_batch(rng, *, batch_size, digits, active_digits=None, carry_heavy_fraction=0.25):
    active_digits = active_digits or digits
    start = digits - active_digits
    a = np.zeros((batch_size, digits), dtype=np.int64)
    b = np.zeros((batch_size, digits), dtype=np.int64)
    a[:, start:] = rng.integers(0, 10, size=(batch_size, active_digits))
    b[:, start:] = rng.integers(0, 10, size=(batch_size, active_digits))
    carry_rows = int(round(batch_size * carry_heavy_fraction))
    if carry_rows:
        sel = rng.choice(batch_size, size=carry_rows, replace=False)
        ca, cb = _carry_chain_operands(rng, carry_rows, active_digits)
        a[sel, start:] = ca
        b[sel, start:] = cb

    answer = np.zeros((batch_size, digits + 1), dtype=np.int64)
    carry = np.zeros(batch_size, dtype=np.int64)
    for position in range(digits - 1, -1, -1):
        total = a[:, position] + b[:, position] + carry
        answer[:, position + 1] = total % 10
        carry = total // 10
    answer[:, 0] = carry

    rows = np.empty((batch_size, sequence_length(digits) + 1), dtype=np.int64)
    rows[:, :digits] = a + DIGIT
    rows[:, digits] = TOKEN_TO_ID["+"]
    rows[:, digits + 1 : 2 * digits + 1] = b + DIGIT
    rows[:, 2 * digits + 1] = TOKEN_TO_ID["="]
    answer_start = 2 * digits + 2
    rows[:, answer_start:-1] = answer[:, ::-1] + DIGIT
    rows[:, -1] = TOKEN_TO_ID[" "]

    inputs = rows[:, :-1].copy()
    targets = rows[:, 1:].copy()
    targets[:, : answer_start - 1] = IGNORE  # loss on answer digits + final space only
    return inputs, targets, answer_start


# --------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, dims, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dims)
        self.qkv = nn.Linear(dims, 3 * dims, bias=False)
        self.proj = nn.Linear(dims, dims, bias=False)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims), nn.GELU(), nn.Linear(4 * dims, dims)
        )
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = (B, T, self.heads, C // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class AdditionTransformer(nn.Module):
    """The 3-digit notebook model, scaled: token + learned absolute positions."""

    def __init__(self, vocab, dims, heads, layers, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, dims)
        self.pos = nn.Embedding(block_size, dims)
        self.blocks = nn.ModuleList(Block(dims, heads, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dims)
        self.head = nn.Linear(dims, vocab)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------- train/eval
def loss_fn(logits, targets):
    return F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]), targets.view(-1), ignore_index=IGNORE
    )


@torch.no_grad()
def probe_eval(model, rng, device, autocast, *, layout, active, batch_size, carry_frac):
    model.eval()
    x, y, _ = make_addition_batch(
        rng, batch_size=batch_size, digits=layout,
        active_digits=active, carry_heavy_fraction=carry_frac,
    )
    x = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    with autocast():
        logits = model(x)
    loss = loss_fn(logits, yt)
    pred = logits.argmax(-1)
    mask = yt != IGNORE
    answer = mask  # loss mask == answer region (answer digits + final space)
    digit_ok = ((pred == yt) & answer).sum() / answer.sum()
    exact = ((pred == yt) | ~answer).all(dim=1).float().mean()
    model.train()
    return {
        "loss": float(loss),
        "answer_digit_accuracy": float(digit_ok),
        "answer_exact_accuracy": float(exact),
    }


@torch.no_grad()
def strict_autoregressive_eval(model, rng, device, autocast, *, layout, active, batch_size):
    """Greedy generation with a fixed-shape buffer (XLA recompilation-safe)."""

    model.eval()
    x, y, answer_start = make_addition_batch(
        rng, batch_size=batch_size, digits=layout,
        active_digits=active, carry_heavy_fraction=0.5,
    )
    buf = torch.from_numpy(x).to(device)  # (B, T) prompt + true answer
    truth = buf.clone()
    T = buf.shape[1]
    for t in range(answer_start, T):  # overwrite answer region with own preds
        with autocast():
            logits = model(buf)
        buf[:, t] = logits[:, t - 1].argmax(-1)
        xla_sync()
    final_target = torch.from_numpy(y[:, -1]).to(device)
    with autocast():
        logits = model(buf)
    last_pred = logits[:, -1].argmax(-1)
    answer_ok = (buf[:, answer_start:] == truth[:, answer_start:]).all(dim=1)
    exact = (answer_ok & (last_pred == final_target)).float().mean()
    model.train()
    return float(exact)


class _Config:
    def __init__(self, values):
        self.__dict__.update(values)
        self.output_dir = Path(self.output_dir)
        self.resume = Path(self.resume) if self.resume else None


def main(config=CONFIG):
    args = _Config(config)

    if XLA:
        device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        autocast_dev = "xla"
    elif torch.cuda.is_available():
        device, autocast_dev = torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        device, autocast_dev = torch.device("mps"), "mps"
    else:
        device, autocast_dev = torch.device("cpu"), "cpu"

    try:
        with torch.autocast(autocast_dev, dtype=torch.bfloat16):
            pass

        def autocast():
            return torch.autocast(autocast_dev, dtype=torch.bfloat16)

    except RuntimeError:  # backend without bf16 autocast (e.g. older MPS)
        import contextlib

        def autocast():
            return contextlib.nullcontext()

    layout = args.digits
    block_size = sequence_length(layout)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    model = AdditionTransformer(
        len(TOKENS), args.dims, args.heads, args.layers, block_size, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_at(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        start_step = state["step"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.jsonl"

    def log(event):
        line = json.dumps(event)
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    n_params = sum(t.numel() for t in model.parameters())
    log({
        "event": "start", "architecture": "full_attention_learned_abs_pos",
        "parameters": n_params, "device": str(device), "xla": XLA,
        "digits": layout, "sequence_length": block_size,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "dims": args.dims, "heads": args.heads, "layers": args.layers,
        "total_steps": args.steps, "resume_step": start_step,
    })

    probes = tuple(w for w in (16, 64, 512) if w < layout) + (layout,)
    model.train()
    window_t, window_n = 0.0, 0
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            if layout > 1 and rng.random() >= args.full_width_fraction:
                active = int(rng.integers(1, layout))
            else:
                active = layout
            x, y, _ = make_addition_batch(
                rng, batch_size=args.batch_size, digits=layout,
                active_digits=active, carry_heavy_fraction=args.carry_heavy_fraction,
            )
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            with autocast():
                logits = model(x)
            loss = loss_fn(logits, y) / args.grad_accum
            loss.backward()
            xla_sync()  # bound peak memory to one microbatch's activations
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, foreach=False
        )
        optimizer.step()
        xla_sync()  # single device: materialize the step; no reduction needed
        window_t += time.perf_counter() - started
        window_n += 1
        done = step + 1

        if done % args.log_every == 0:
            log({
                "event": "train", "step": done, "active_digits": active,
                "loss": float(loss) * args.grad_accum, "gradient_norm": float(grad_norm),
                "learning_rate": lr_at(step), "step_ms": 1000 * window_t / window_n,
            })
            window_t, window_n = 0.0, 0

        if done % args.eval_every == 0:
            report = {}
            for width in probes:
                report[str(width)] = {
                    "random": probe_eval(
                        model, eval_rng, device, autocast, layout=layout,
                        active=width, batch_size=args.eval_batch_size, carry_frac=0.0,
                    ),
                    "carry_chain": probe_eval(
                        model, eval_rng, device, autocast, layout=layout,
                        active=width, batch_size=args.eval_batch_size, carry_frac=1.0,
                    ),
                }
            log({"event": "validation", "step": done, "fixed_layout": True, "probes": report})

        if args.strict_eval_every and done % args.strict_eval_every == 0:
            exact = strict_autoregressive_eval(
                model, eval_rng, device, autocast,
                layout=layout, active=layout, batch_size=8,
            )
            log({"event": "strict_autoregressive", "step": done, "exact_accuracy": exact})

        if done % args.checkpoint_every == 0 or done == args.steps:
            path = args.output_dir / f"checkpoint-{done:08d}.pt"
            payload = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng.bit_generator.state, "step": done, "config": vars(args) | {"output_dir": str(args.output_dir), "resume": None},
            }
            if XLA:
                xm.save(payload, path)
            else:
                torch.save(payload, path)
            (args.output_dir / "latest.json").write_text(json.dumps({"checkpoint": path.name}) + "\n")
            log({"event": "checkpoint", "path": str(path)})

    log({"event": "complete", "step": args.steps})


if __name__ == "__main__":
    main()
