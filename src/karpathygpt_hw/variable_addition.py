"""Variable-length, unpadded addition Transformer experiments.

This module intentionally keeps the model close to ``notebooks/gpt-hw.ipynb``:

* character-level tokenization;
* learned absolute position embeddings;
* dense causal multi-head self-attention;
* four-times-wide ReLU feed-forward layers;
* pre-norm residual blocks; and
* AdamW optimization with answer-only next-token loss.

The data representation is the experimental variable. Operands have no leading
zeroes; answers can be reversed or normal-order without width padding; and every
batch contains one exact ``(digits_a, digits_b, digits_answer)`` layout. Consequently,
tensors do not contain padding tokens even though sequence lengths vary between batches.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

TOKENS = (" ", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=")
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
EOS_TOKEN = " "
IGNORE_INDEX = -1
AnswerOrder = Literal["reversed", "normal"]


@dataclass(frozen=True, order=True)
class AdditionLayout:
    """The number of decimal digits in both operands and their answer."""

    digits_a: int
    digits_b: int
    digits_answer: int

    @property
    def prompt_length(self) -> int:
        return self.digits_a + self.digits_b + 2

    @property
    def row_length(self) -> int:
        # A + B = reversed(answer) + EOS
        return self.prompt_length + self.digits_answer + 1

    @property
    def model_input_length(self) -> int:
        return self.row_length - 1


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int

    @property
    def answer(self) -> int:
        return self.a + self.b


@dataclass(frozen=True)
class AdditionBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    layout: AdditionLayout
    examples: tuple[AdditionExample, ...]


@dataclass(frozen=True)
class EvalGroup:
    name: str
    layout: AdditionLayout
    examples: tuple[AdditionExample, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    max_digits: int
    output_dir: Path
    answer_order: AnswerOrder = "reversed"
    steps: int = 5_000
    batch_size: int = 256
    dims: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    hard_carry_fraction: float = 0.25
    eval_every: int = 250
    log_every: int = 50
    quick_eval_cases: int = 128
    final_eval_cases: int = 256
    seed: int = 4_242
    device: str = "auto"

    @property
    def block_size(self) -> int:
        # Maximum shifted-input length for D+D=(D+1 digits), including EOS target.
        return 3 * self.max_digits + 3


def encode(text: str) -> list[int]:
    return [TOKEN_TO_ID[character] for character in text]


def decode(token_ids: Iterable[int]) -> str:
    return "".join(ID_TO_TOKEN[int(token)] for token in token_ids)


def decimal_digits(value: int) -> int:
    if value < 0:
        raise ValueError("decimal_digits expects a non-negative integer")
    return 1 if value == 0 else len(str(value))


def digit_bounds(digits: int) -> tuple[int, int]:
    if digits < 1:
        raise ValueError("digits must be positive")
    if digits == 1:
        return 0, 9
    return 10 ** (digits - 1), 10**digits - 1


def valid_layouts(max_digits: int) -> tuple[AdditionLayout, ...]:
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    layouts: list[AdditionLayout] = []
    for digits_a in range(1, max_digits + 1):
        a_low, a_high = digit_bounds(digits_a)
        for digits_b in range(1, max_digits + 1):
            b_low, b_high = digit_bounds(digits_b)
            min_sum = a_low + b_low
            max_sum = a_high + b_high
            for digits_answer in range(1, max_digits + 2):
                c_low, c_high = digit_bounds(digits_answer)
                if max(min_sum, c_low) <= min(max_sum, c_high):
                    layouts.append(AdditionLayout(digits_a, digits_b, digits_answer))
    return tuple(layouts)


def _split_bucket(a: int, b: int, seed: int) -> int:
    """Stable SplitMix64-style pair hash, returned modulo 100."""

    mask = (1 << 64) - 1

    def mix(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    value = mix(a + seed * 0x100000001B3) ^ mix(b + seed * 0x9E3779B1)
    value = mix(value)
    return int(value % 100)


def pair_split(a: int, b: int, seed: int) -> str:
    bucket = _split_bucket(a, b, seed)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _sample_layout_pair(
    rng: np.random.Generator,
    layout: AdditionLayout,
    *,
    split: str,
    split_seed: int,
) -> AdditionExample:
    """Sample a pair conditioned on exact operand and answer digit lengths."""

    a_low, a_high = digit_bounds(layout.digits_a)
    b_low, b_high = digit_bounds(layout.digits_b)
    c_low, c_high = digit_bounds(layout.digits_answer)

    valid_a_low = max(a_low, c_low - b_high)
    valid_a_high = min(a_high, c_high - b_low)
    if valid_a_low > valid_a_high:
        raise ValueError(f"infeasible layout: {layout}")

    for _ in range(100_000):
        a = int(rng.integers(valid_a_low, valid_a_high + 1))
        valid_b_low = max(b_low, c_low - a)
        valid_b_high = min(b_high, c_high - a)
        if valid_b_low > valid_b_high:
            continue
        b = int(rng.integers(valid_b_low, valid_b_high + 1))
        if pair_split(a, b, split_seed) == split:
            return AdditionExample(a, b)
    raise RuntimeError(f"could not sample {split} pair for layout {layout}")


def carry_chain_length(a: int, b: int) -> int:
    """Return the longest consecutive decimal carry chain."""

    carry = 0
    current = 0
    longest = 0
    while a or b or carry:
        total = a % 10 + b % 10 + carry
        carry = int(total >= 10)
        if carry:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        a //= 10
        b //= 10
    return longest


def _sample_full_carry_pair(
    rng: np.random.Generator,
    digits: int,
    *,
    split: str,
    split_seed: int,
) -> AdditionExample:
    """Sample exact-width operands carrying through all columns and out the top."""

    for _ in range(100_000):
        a_digits = rng.integers(0, 10, size=digits, dtype=np.int64)
        a_digits[0] = rng.integers(1, 10)
        a_digits[-1] = rng.integers(1, 10)
        b_digits = np.empty(digits, dtype=np.int64)
        for position in range(digits - 1, -1, -1):
            minimum = (
                10 - int(a_digits[position])
                if position == digits - 1
                else 9 - int(a_digits[position])
            )
            if position == 0:
                minimum = max(1, minimum)
            minimum = max(0, minimum)
            b_digits[position] = rng.integers(minimum, 10)
        a = int("".join(str(int(digit)) for digit in a_digits))
        b = int("".join(str(int(digit)) for digit in b_digits))
        if pair_split(a, b, split_seed) != split:
            continue
        example = AdditionExample(a, b)
        if decimal_digits(example.answer) != digits + 1:
            continue
        if carry_chain_length(a, b) == digits:
            return example
    raise RuntimeError(f"could not sample full-carry {split} pair at {digits} digits")


def format_answer(example: AdditionExample, answer_order: AnswerOrder = "reversed") -> str:
    answer = str(example.answer)
    if answer_order == "reversed":
        return answer[::-1]
    if answer_order == "normal":
        return answer
    raise ValueError(f"unsupported answer order: {answer_order}")


def format_row(example: AdditionExample, answer_order: AnswerOrder = "reversed") -> str:
    return f"{example.a}+{example.b}={format_answer(example, answer_order)}{EOS_TOKEN}"


def example_layout(example: AdditionExample) -> AdditionLayout:
    return AdditionLayout(
        decimal_digits(example.a),
        decimal_digits(example.b),
        decimal_digits(example.answer),
    )


def make_batch(
    examples: Iterable[AdditionExample],
    *,
    device: torch.device,
    answer_order: AnswerOrder = "reversed",
) -> AdditionBatch:
    examples_tuple = tuple(examples)
    if not examples_tuple:
        raise ValueError("cannot make an empty batch")
    layout = example_layout(examples_tuple[0])
    if any(example_layout(example) != layout for example in examples_tuple):
        raise ValueError("all examples in an unpadded batch must share one exact layout")

    rows = [encode(format_row(example, answer_order)) for example in examples_tuple]
    expected_length = layout.row_length
    if any(len(row) != expected_length for row in rows):
        raise AssertionError("layout and encoded row length disagree")
    full = torch.tensor(rows, dtype=torch.long, device=device)
    inputs = full[:, :-1]
    targets = full[:, 1:].clone()
    # The target at prompt_length-1 is the first answer digit predicted from '='.
    targets[:, : layout.prompt_length - 1] = IGNORE_INDEX
    return AdditionBatch(inputs, targets, layout, examples_tuple)


class DenseCausalSelfAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if dims % num_heads:
            raise ValueError("dims must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dims = dims // num_heads
        self.key = nn.Linear(dims, dims, bias=False)
        self.query = nn.Linear(dims, dims, bias=False)
        self.value = nn.Linear(dims, dims, bias=False)
        self.projection = nn.Linear(dims, dims)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, dims = inputs.shape
        keys = self.key(inputs).view(batch, length, self.num_heads, self.head_dims).transpose(1, 2)
        queries = (
            self.query(inputs).view(batch, length, self.num_heads, self.head_dims).transpose(1, 2)
        )
        values = (
            self.value(inputs).view(batch, length, self.num_heads, self.head_dims).transpose(1, 2)
        )
        scores = queries @ keys.transpose(-2, -1) * (self.head_dims**-0.5)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        output = weights @ values
        output = output.transpose(1, 2).contiguous().view(batch, length, dims)
        return self.residual_dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, dims: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims, 4 * dims),
            nn.ReLU(),
            nn.Linear(4 * dims, dims),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.attention = DenseCausalSelfAttention(dims, num_heads, block_size, dropout)
        self.feed_forward = FeedForward(dims, dropout)
        self.norm_attention = nn.LayerNorm(dims)
        self.norm_feed_forward = nn.LayerNorm(dims)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.norm_attention(inputs))
        return hidden + self.feed_forward(self.norm_feed_forward(hidden))


class AdditionTransformer(nn.Module):
    """The original notebook architecture with runtime-configurable maximum length."""

    def __init__(
        self,
        *,
        block_size: int,
        dims: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embeddings = nn.Embedding(len(TOKENS), dims)
        self.position_embeddings = nn.Embedding(block_size, dims)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dims, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dims)
        self.lm_head = nn.Linear(dims, len(TOKENS))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.block_size:
            raise ValueError(f"input length {length} exceeds block size {self.block_size}")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embeddings(token_ids) + self.position_embeddings(positions)
        hidden = self.blocks(hidden)
        return self.lm_head(self.final_norm(hidden))


def masked_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    metadata["block_size"] = config.block_size
    return metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def sample_training_batch(
    rng: np.random.Generator,
    layouts: tuple[AdditionLayout, ...],
    config: ExperimentConfig,
    device: torch.device,
) -> AdditionBatch:
    hard_carry = bool(rng.random() < config.hard_carry_fraction)
    if hard_carry:
        digits = int(rng.integers(1, config.max_digits + 1))
        examples = [
            _sample_full_carry_pair(rng, digits, split="train", split_seed=config.seed)
            for _ in range(config.batch_size)
        ]
    else:
        layout = layouts[int(rng.integers(0, len(layouts)))]
        examples = [
            _sample_layout_pair(rng, layout, split="train", split_seed=config.seed)
            for _ in range(config.batch_size)
        ]
    return make_batch(examples, device=device, answer_order=config.answer_order)


def _enumerated_eval_candidates(
    layout: AdditionLayout,
    *,
    split: str,
    split_seed: int,
) -> list[AdditionExample] | None:
    a_low, a_high = digit_bounds(layout.digits_a)
    b_low, b_high = digit_bounds(layout.digits_b)
    pair_count = (a_high - a_low + 1) * (b_high - b_low + 1)
    if pair_count > 200_000:
        return None
    candidates = []
    for a in range(a_low, a_high + 1):
        for b in range(b_low, b_high + 1):
            if decimal_digits(a + b) != layout.digits_answer:
                continue
            if pair_split(a, b, split_seed) == split:
                candidates.append(AdditionExample(a, b))
    return candidates


def sample_eval_examples(
    rng: np.random.Generator,
    layout: AdditionLayout,
    *,
    count: int,
    split: str,
    split_seed: int,
) -> tuple[AdditionExample, ...]:
    enumerated = _enumerated_eval_candidates(layout, split=split, split_seed=split_seed)
    if enumerated is not None:
        rng.shuffle(enumerated)
        return tuple(enumerated[:count])

    selected: dict[tuple[int, int], AdditionExample] = {}
    max_attempts = max(10_000, count * 1_000)
    for _ in range(max_attempts):
        example = _sample_layout_pair(rng, layout, split=split, split_seed=split_seed)
        selected[(example.a, example.b)] = example
        if len(selected) >= count:
            break
    if not selected:
        raise RuntimeError(f"no evaluation examples for {layout}")
    return tuple(selected.values())


def build_eval_groups(
    *,
    max_digits: int,
    cases_per_layout: int,
    split: str,
    seed: int,
    quick: bool,
) -> tuple[EvalGroup, ...]:
    rng = np.random.default_rng(seed)
    layouts = valid_layouts(max_digits)
    if quick:
        layouts = tuple(
            layout
            for layout in layouts
            if layout.digits_a == max_digits and layout.digits_b == max_digits
        )
    groups: list[EvalGroup] = []
    for layout in layouts:
        examples = sample_eval_examples(
            rng,
            layout,
            count=cases_per_layout,
            split=split,
            split_seed=seed,
        )
        name = f"layout-{layout.digits_a}-{layout.digits_b}-{layout.digits_answer}"
        groups.append(EvalGroup(name, layout, examples))

    # A separately-labelled worst-case set makes long carry failures visible.
    hard_digits = (max_digits,) if quick else tuple(range(1, max_digits + 1))
    for digits in hard_digits:
        lower, upper = digit_bounds(digits)
        if (upper - lower + 1) ** 2 <= 200_000:
            candidates = [
                AdditionExample(a, b)
                for a in range(lower, upper + 1)
                for b in range(lower, upper + 1)
                if decimal_digits(a + b) == digits + 1
                and carry_chain_length(a, b) == digits
                and pair_split(a, b, seed) == split
            ]
            rng.shuffle(candidates)
            hard_examples = tuple(candidates[:cases_per_layout])
        else:
            examples_by_pair: dict[tuple[int, int], AdditionExample] = {}
            for _ in range(max(10_000, cases_per_layout * 1_000)):
                example = _sample_full_carry_pair(
                    rng,
                    digits,
                    split=split,
                    split_seed=seed,
                )
                examples_by_pair[(example.a, example.b)] = example
                if len(examples_by_pair) >= cases_per_layout:
                    break
            hard_examples = tuple(examples_by_pair.values())
        if not hard_examples:
            raise RuntimeError(f"no hard-carry {split} examples at {digits} digits")
        groups.append(
            EvalGroup(
                f"hard-carry-{digits}",
                AdditionLayout(digits, digits, digits + 1),
                hard_examples,
            )
        )
    return tuple(groups)


@torch.inference_mode()
def evaluate_group(
    model: AdditionTransformer,
    group: EvalGroup,
    device: torch.device,
    answer_order: AnswerOrder = "reversed",
) -> dict[str, Any]:
    batch = make_batch(group.examples, device=device, answer_order=answer_order)
    logits = model(batch.inputs)
    loss = masked_loss(logits, batch.targets)

    prompts = [f"{example.a}+{example.b}=" for example in group.examples]
    prompt_tokens = torch.tensor([encode(prompt) for prompt in prompts], device=device)
    generated = prompt_tokens
    for _ in range(group.layout.digits_answer + 1):
        next_logits = model(generated)[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    continuation = generated[:, group.layout.prompt_length :]
    expected_text = [format_answer(example, answer_order) + EOS_TOKEN for example in group.examples]
    expected = torch.tensor([encode(text) for text in expected_text], device=device)
    exact = torch.all(continuation == expected, dim=1)
    answer_exact = torch.all(
        continuation[:, : group.layout.digits_answer] == expected[:, : group.layout.digits_answer],
        dim=1,
    )
    digit_correct = (
        continuation[:, : group.layout.digits_answer] == expected[:, : group.layout.digits_answer]
    )
    eos_correct = continuation[:, -1] == expected[:, -1]
    predictions = [decode(row.tolist()) for row in continuation.cpu()]
    return {
        "name": group.name,
        "layout": asdict(group.layout),
        "cases": len(group.examples),
        "loss": float(loss.item()),
        "sequence_exact_accuracy": float(exact.float().mean().item()),
        "answer_exact_accuracy": float(answer_exact.float().mean().item()),
        "answer_digit_accuracy": float(digit_correct.float().mean().item()),
        "eos_accuracy": float(eos_correct.float().mean().item()),
        "correct_answers": int(answer_exact.sum().item()),
        "correct_digits": int(digit_correct.sum().item()),
        "answer_digits": int(digit_correct.numel()),
        "examples": [
            {
                "a": example.a,
                "b": example.b,
                "expected_answer": expected_value.rstrip(EOS_TOKEN),
                "predicted_answer": predicted.rstrip(EOS_TOKEN),
                "correct": predicted.rstrip(EOS_TOKEN) == expected_value.rstrip(EOS_TOKEN),
                "carry_chain": carry_chain_length(example.a, example.b),
            }
            for example, expected_value, predicted in zip(
                group.examples[:8], expected_text[:8], predictions[:8], strict=True
            )
        ],
    }


@torch.inference_mode()
def evaluate_groups(
    model: AdditionTransformer,
    groups: tuple[EvalGroup, ...],
    device: torch.device,
    answer_order: AnswerOrder = "reversed",
) -> dict[str, Any]:
    model.eval()
    group_metrics = [evaluate_group(model, group, device, answer_order) for group in groups]
    total_cases = sum(metric["cases"] for metric in group_metrics)
    total_correct = sum(metric["correct_answers"] for metric in group_metrics)
    total_digits = sum(metric["answer_digits"] for metric in group_metrics)
    total_correct_digits = sum(metric["correct_digits"] for metric in group_metrics)
    layout_metrics = [metric for metric in group_metrics if metric["name"].startswith("layout-")]
    hard_metrics = [metric for metric in group_metrics if metric["name"].startswith("hard-carry-")]
    result = {
        "cases": total_cases,
        "micro_answer_exact_accuracy": total_correct / total_cases if total_cases else 0.0,
        "micro_answer_digit_accuracy": total_correct_digits / total_digits if total_digits else 0.0,
        "macro_layout_exact_accuracy": float(
            np.mean([metric["answer_exact_accuracy"] for metric in layout_metrics])
        )
        if layout_metrics
        else None,
        "macro_hard_carry_exact_accuracy": float(
            np.mean([metric["answer_exact_accuracy"] for metric in hard_metrics])
        )
        if hard_metrics
        else None,
        "groups": group_metrics,
    }
    model.train()
    return result


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def train(config: ExperimentConfig) -> dict[str, Any]:
    if config.max_digits < 1:
        raise ValueError("max_digits must be positive")
    if not 0 <= config.hard_carry_fraction <= 1:
        raise ValueError("hard_carry_fraction must be between zero and one")
    if config.answer_order not in {"reversed", "normal"}:
        raise ValueError(f"unsupported answer order: {config.answer_order}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    _write_json(config.output_dir / "config.json", _jsonable_config(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    layouts = valid_layouts(config.max_digits)

    model = AdditionTransformer(
        block_size=config.block_size,
        dims=config.dims,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    quick_groups = build_eval_groups(
        max_digits=config.max_digits,
        cases_per_layout=config.quick_eval_cases,
        split="val",
        seed=config.seed,
        quick=True,
    )

    metadata = {
        **_jsonable_config(config),
        "event": "start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": "dense_causal_attention_learned_absolute_positions",
        "data": f"variable_length_unpadded_{config.answer_order}_answer",
        "parameters": parameter_count(model),
        "device": str(device),
        "requested_device": config.device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "layout_count": len(layouts),
    }
    _append_event(metrics_path, metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)

    _synchronize(device)
    train_started = time.perf_counter()
    interval_started = train_started
    interval_examples = 0
    interval_tokens = 0
    last_loss = math.nan
    peak_memory = _peak_memory_bytes(device) or 0

    model.train()
    for step in range(1, config.steps + 1):
        batch = sample_training_batch(rng, layouts, config, device)
        logits = model(batch.inputs)
        loss = masked_loss(logits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        interval_examples += len(batch.examples)
        interval_tokens += batch.inputs.numel()

        if step % config.log_every == 0 or step == 1:
            _synchronize(device)
            now = time.perf_counter()
            elapsed = now - interval_started
            peak_memory = max(peak_memory, _peak_memory_bytes(device) or 0)
            event = {
                "event": "train",
                "step": step,
                "loss": last_loss,
                "layout": asdict(batch.layout),
                "examples_per_second": interval_examples / elapsed,
                "tokens_per_second": interval_tokens / elapsed,
                "elapsed_training_seconds": now - train_started,
                "driver_allocated_bytes": _peak_memory_bytes(device),
            }
            _append_event(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            interval_started = now
            interval_examples = 0
            interval_tokens = 0

        if step % config.eval_every == 0 or step == config.steps:
            _synchronize(device)
            eval_started = time.perf_counter()
            validation = evaluate_groups(model, quick_groups, device, config.answer_order)
            _synchronize(device)
            event = {
                "event": "validation",
                "step": step,
                "elapsed_seconds": time.perf_counter() - eval_started,
                **{key: value for key, value in validation.items() if key != "groups"},
                "groups": validation["groups"],
            }
            _append_event(metrics_path, event)
            print(
                json.dumps(
                    {key: value for key, value in event.items() if key not in {"groups"}},
                    sort_keys=True,
                ),
                flush=True,
            )

    _synchronize(device)
    training_seconds = time.perf_counter() - train_started
    checkpoint = {
        "config": _jsonable_config(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "parameters": parameter_count(model),
        "steps": config.steps,
    }
    torch.save(checkpoint, config.output_dir / "model.pt")

    final_groups = build_eval_groups(
        max_digits=config.max_digits,
        cases_per_layout=config.final_eval_cases,
        split="test",
        seed=config.seed,
        quick=False,
    )
    _synchronize(device)
    final_eval_started = time.perf_counter()
    final_metrics = evaluate_groups(model, final_groups, device, config.answer_order)
    _synchronize(device)
    final_eval_seconds = time.perf_counter() - final_eval_started
    peak_memory = max(peak_memory, _peak_memory_bytes(device) or 0)
    result = {
        "event": "complete",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_digits": config.max_digits,
        "parameters": parameter_count(model),
        "steps": config.steps,
        "final_training_loss": last_loss,
        "training_seconds": training_seconds,
        "final_evaluation_seconds": final_eval_seconds,
        "total_seconds": training_seconds + final_eval_seconds,
        "peak_driver_allocated_bytes": peak_memory or None,
        "test": final_metrics,
    }
    _write_json(config.output_dir / "final_metrics.json", result)
    _append_event(metrics_path, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "test"}
            | {
                "test_answer_exact": final_metrics["micro_answer_exact_accuracy"],
                "test_hard_carry_exact": final_metrics["macro_hard_carry_exact_accuracy"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-digits", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-order", choices=("reversed", "normal"), default="reversed")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hard-carry-fraction", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--quick-eval-cases", type=int, default=128)
    parser.add_argument("--final-eval-cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4_242)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(ExperimentConfig(**vars(args)))


if __name__ == "__main__":
    main()
