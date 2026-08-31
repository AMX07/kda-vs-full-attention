"""Curriculum training utilities for long, character-level addition with KDA."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from karpathygpt_hw.kda_mlx import KDAAdditionModel, masked_addition_loss, parameter_count

TOKENS = (" ", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=")
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
DIGIT_TOKEN_OFFSET = TOKEN_TO_ID["0"]
IGNORE_INDEX = -1


@dataclass(frozen=True)
class CurriculumStage:
    digits: int
    steps: int


SMOKE_SCHEDULE = (
    CurriculumStage(3, 20),
    CurriculumStage(8, 20),
)

# A conservative first long-context experiment. This is a starting schedule,
# not a claim that 60K steps is sufficient for 1,364-digit generalization.
FULL_SCHEDULE = (
    CurriculumStage(3, 2_000),
    CurriculumStage(8, 3_000),
    CurriculumStage(16, 4_000),
    CurriculumStage(32, 5_000),
    CurriculumStage(64, 6_000),
    CurriculumStage(128, 7_000),
    CurriculumStage(256, 8_000),
    CurriculumStage(512, 8_000),
    CurriculumStage(1_024, 8_000),
    CurriculumStage(1_364, 9_000),
)


@dataclass(frozen=True)
class AdditionBatch:
    inputs: np.ndarray
    targets: np.ndarray
    digits: int
    active_digits: int

    @property
    def answer_slice(self) -> slice:
        start = 2 * self.digits + 1
        return slice(start, start + self.digits + 1)


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    answer_digit_accuracy: float
    answer_exact_accuracy: float
    trailing_space_accuracy: float


@dataclass(frozen=True)
class TrainConfig:
    schedule: tuple[CurriculumStage, ...]
    output_dir: Path
    batch_size: int = 64
    eval_batch_size: int = 64
    dims: int = 64
    num_heads: int = 4
    num_layers: int = 4
    position_encoding: str = "none"
    decay_half_life_min: float | None = None
    decay_half_life_max: float | None = None
    learning_rate: float = 1e-3
    lr_decay_steps: int | None = None
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    max_grad_norm: float = 1.0
    carry_heavy_fraction: float = 0.25
    rehearsal_fraction: float = 0.10
    rehearsal_mode: str = "padded"
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 1_000
    seed: int = 4_242

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.schedule)


def sequence_length(digits: int) -> int:
    """Model-input length for ``aaa+bbb=cccc`` with a shifted target."""

    if digits < 1:
        raise ValueError("digits must be positive")
    return 3 * digits + 3


def parse_schedule(value: str) -> tuple[CurriculumStage, ...]:
    """Parse ``DIGITS:STEPS`` pairs such as ``3:500,8:1000``."""

    stages = []
    for item in value.split(","):
        try:
            digits_text, steps_text = item.strip().split(":", maxsplit=1)
            stage = CurriculumStage(int(digits_text), int(steps_text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid schedule item {item!r}; expected DIGITS:STEPS") from error
        if stage.digits < 1 or stage.steps < 1:
            raise ValueError("schedule digits and steps must be positive")
        if stages and stage.digits <= stages[-1].digits:
            raise ValueError("curriculum digit widths must be strictly increasing")
        stages.append(stage)
    if not stages:
        raise ValueError("schedule must contain at least one stage")
    if stages[-1].digits > 1_364:
        raise ValueError("the current 4,096-character format supports at most 1,364 digits")
    return tuple(stages)


def _carry_chain_operands(
    rng: np.random.Generator,
    rows: int,
    digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate operands that carry through every active digit column."""

    a = np.zeros((rows, digits), dtype=np.int32)
    b = np.zeros((rows, digits), dtype=np.int32)
    for position in range(digits - 1, -1, -1):
        if position == digits - 1:
            a_column = rng.integers(1, 10, size=rows, dtype=np.int32)
            minimum_b = 10 - a_column
        else:
            a_column = rng.integers(0, 10, size=rows, dtype=np.int32)
            minimum_b = np.maximum(0, 9 - a_column)
        span = 10 - minimum_b
        b_column = minimum_b + (rng.random(rows) * span).astype(np.int32)
        a[:, position] = a_column
        b[:, position] = b_column
    return a, b


def make_addition_batch(
    rng: np.random.Generator,
    *,
    batch_size: int,
    digits: int,
    active_digits: int | None = None,
    carry_heavy_fraction: float = 0.25,
) -> AdditionBatch:
    """Create fixed-width addition rows without converting through Python integers.

    The complete row is ``A+B=reverse(zfill(A+B, digits+1))<space>``. Inputs
    omit the final space and targets are shifted by one. Targets belonging to
    the prompt are masked with ``-1`` so only answer digits and the final space
    contribute to the language-modeling loss.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if digits < 1:
        raise ValueError("digits must be positive")
    active_digits = active_digits or digits
    if not 1 <= active_digits <= digits:
        raise ValueError("active_digits must be between 1 and digits")
    if not 0.0 <= carry_heavy_fraction <= 1.0:
        raise ValueError("carry_heavy_fraction must be between 0 and 1")

    start = digits - active_digits
    a = np.zeros((batch_size, digits), dtype=np.int32)
    b = np.zeros((batch_size, digits), dtype=np.int32)
    a[:, start:] = rng.integers(0, 10, size=(batch_size, active_digits), dtype=np.int32)
    b[:, start:] = rng.integers(0, 10, size=(batch_size, active_digits), dtype=np.int32)

    carry_rows = int(round(batch_size * carry_heavy_fraction))
    if carry_rows:
        selected = rng.choice(batch_size, size=carry_rows, replace=False)
        carry_a, carry_b = _carry_chain_operands(rng, carry_rows, active_digits)
        a[selected, start:] = carry_a
        b[selected, start:] = carry_b

    answer = np.zeros((batch_size, digits + 1), dtype=np.int32)
    carry = np.zeros(batch_size, dtype=np.int32)
    for position in range(digits - 1, -1, -1):
        total = a[:, position] + b[:, position] + carry
        answer[:, position + 1] = total % 10
        carry = total // 10
    answer[:, 0] = carry

    full_length = sequence_length(digits) + 1
    rows = np.empty((batch_size, full_length), dtype=np.int32)
    rows[:, :digits] = a + DIGIT_TOKEN_OFFSET
    rows[:, digits] = TOKEN_TO_ID["+"]
    rows[:, digits + 1 : 2 * digits + 1] = b + DIGIT_TOKEN_OFFSET
    rows[:, 2 * digits + 1] = TOKEN_TO_ID["="]
    answer_start = 2 * digits + 2
    rows[:, answer_start : answer_start + digits + 1] = answer[:, ::-1] + DIGIT_TOKEN_OFFSET
    rows[:, -1] = TOKEN_TO_ID[" "]

    inputs = rows[:, :-1].copy()
    targets = rows[:, 1:].copy()
    targets[:, : 2 * digits + 1] = IGNORE_INDEX
    return AdditionBatch(inputs, targets, digits, active_digits)


def decode_tokens(token_ids: np.ndarray) -> str:
    return "".join(TOKENS[int(token)] for token in token_ids)


def _learning_rate_schedule(config: TrainConfig) -> Any:
    schedule_steps = config.lr_decay_steps or config.total_steps
    warmup = min(config.warmup_steps, max(1, schedule_steps // 10))
    if warmup >= schedule_steps:
        return optim.linear_schedule(
            config.learning_rate * 0.1,
            config.learning_rate,
            schedule_steps,
        )
    return optim.join_schedules(
        [
            optim.linear_schedule(config.learning_rate * 0.1, config.learning_rate, warmup),
            optim.cosine_decay(
                config.learning_rate,
                schedule_steps - warmup,
                end=config.learning_rate * 0.1,
            ),
        ],
        boundaries=[warmup],
    )


def _compile_train_step(
    model: KDAAdditionModel,
    optimizer: optim.Optimizer,
    max_grad_norm: float,
) -> Any:
    value_and_grad = nn.value_and_grad(model, masked_addition_loss)

    def train_step(inputs: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
        loss, gradients = value_and_grad(model, inputs, targets)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, max_grad_norm)
        optimizer.update(model, gradients)
        return loss, gradient_norm

    return mx.compile(
        train_step,
        inputs=[model.state, optimizer.state],
        outputs=[model.state, optimizer.state],
    )


def evaluate(
    model: KDAAdditionModel,
    rng: np.random.Generator,
    *,
    batch_size: int,
    digits: int,
    carry_heavy_fraction: float,
) -> ValidationMetrics:
    batch = make_addition_batch(
        rng,
        batch_size=batch_size,
        digits=digits,
        carry_heavy_fraction=carry_heavy_fraction,
    )
    model.eval()
    inputs = mx.array(batch.inputs)
    targets = mx.array(batch.targets)
    logits = model(inputs)
    mask = targets != IGNORE_INDEX
    safe_targets = mx.where(mask, targets, 0)
    token_losses = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    loss = mx.sum(mx.where(mask, token_losses, 0)) / mx.sum(mask)
    predictions = mx.argmax(logits, axis=-1)
    mx.eval(loss, predictions)
    predicted = np.asarray(predictions)
    answer_slice = batch.answer_slice
    expected_answer = batch.targets[:, answer_slice]
    predicted_answer = predicted[:, answer_slice]
    metrics = ValidationMetrics(
        loss=float(loss),
        answer_digit_accuracy=float(np.mean(predicted_answer == expected_answer)),
        answer_exact_accuracy=float(np.mean(np.all(predicted_answer == expected_answer, axis=1))),
        trailing_space_accuracy=float(np.mean(predicted[:, -1] == batch.targets[:, -1])),
    )
    model.train()
    return metrics


def _config_metadata(config: TrainConfig) -> dict[str, Any]:
    return {
        "schedule": [asdict(stage) for stage in config.schedule],
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "dims": config.dims,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "position_encoding": config.position_encoding,
        "decay_half_life_min": config.decay_half_life_min,
        "decay_half_life_max": config.decay_half_life_max,
        "learning_rate": config.learning_rate,
        "lr_decay_steps": config.lr_decay_steps,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "max_grad_norm": config.max_grad_norm,
        "carry_heavy_fraction": config.carry_heavy_fraction,
        "rehearsal_fraction": config.rehearsal_fraction,
        "rehearsal_mode": config.rehearsal_mode,
        "seed": config.seed,
    }


def save_checkpoint(
    model: KDAAdditionModel,
    optimizer: optim.Optimizer,
    rng: np.random.Generator,
    config: TrainConfig,
    *,
    global_step: int,
    stage_index: int,
    stage_step: int,
) -> Path:
    """Atomically save model/optimizer arrays and exact curriculum position."""

    checkpoint_dir = config.output_dir / f"checkpoint-{global_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(model.parameters(), optimizer.state)

    model_temp = checkpoint_dir / "model.tmp.safetensors"
    model_path = checkpoint_dir / "model.safetensors"
    optimizer_temp = checkpoint_dir / "optimizer.tmp.safetensors"
    optimizer_path = checkpoint_dir / "optimizer.safetensors"
    state_temp = checkpoint_dir / "state.tmp.json"
    state_path = checkpoint_dir / "state.json"

    model.save_weights(str(model_temp))
    mx.save_safetensors(optimizer_temp, dict(tree_flatten(optimizer.state)))
    state = {
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "rng_state": rng.bit_generator.state,
        "config": _config_metadata(config),
    }
    state_temp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(model_temp, model_path)
    os.replace(optimizer_temp, optimizer_path)
    os.replace(state_temp, state_path)

    latest_temp = config.output_dir / "latest.tmp.json"
    latest_path = config.output_dir / "latest.json"
    latest_temp.write_text(json.dumps({"checkpoint": checkpoint_dir.name}) + "\n")
    os.replace(latest_temp, latest_path)
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    model: KDAAdditionModel,
    optimizer: optim.Optimizer,
    rng: np.random.Generator,
    config: TrainConfig,
) -> tuple[int, int, int]:
    state = json.loads((checkpoint_dir / "state.json").read_text())
    saved_config = state["config"]
    current_config = _config_metadata(config)
    for key in (
        "schedule",
        "dims",
        "num_heads",
        "num_layers",
        "position_encoding",
        "decay_half_life_min",
        "decay_half_life_max",
        "lr_decay_steps",
        "rehearsal_mode",
    ):
        saved_value = (
            saved_config.get(key, "none") if key == "position_encoding" else saved_config[key]
        )
        if saved_value != current_config[key]:
            raise ValueError(f"resume configuration mismatch for {key}")

    model.load_weights(str(checkpoint_dir / "model.safetensors"))
    optimizer_arrays = mx.load(str(checkpoint_dir / "optimizer.safetensors"))
    optimizer.state = tree_unflatten(list(optimizer_arrays.items()))
    rng.bit_generator.state = state["rng_state"]
    mx.eval(model.parameters(), optimizer.state)
    return int(state["global_step"]), int(state["stage_index"]), int(state["stage_step"])


def load_fork_checkpoint(
    checkpoint_dir: Path,
    model: KDAAdditionModel,
    optimizer: optim.Optimizer,
    rng: np.random.Generator,
) -> int:
    """Load learned state while starting a new curriculum branch.

    Unlike an exact resume, a fork intentionally permits a new curriculum and
    output directory. Model weights, AdamW moments/step, and RNG state are kept.
    """

    state = json.loads((checkpoint_dir / "state.json").read_text())
    model.load_weights(str(checkpoint_dir / "model.safetensors"))
    optimizer_arrays = mx.load(str(checkpoint_dir / "optimizer.safetensors"))
    optimizer.state = tree_unflatten(list(optimizer_arrays.items()))
    rng.bit_generator.state = state["rng_state"]
    mx.eval(model.parameters(), optimizer.state)
    return int(state["global_step"])


def train(
    config: TrainConfig,
    *,
    resume: Path | None = None,
    fork_from: Path | None = None,
) -> None:
    if not mx.metal.is_available():
        raise RuntimeError("long-addition KDA training requires Apple Metal")
    if config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if not 0.0 <= config.carry_heavy_fraction <= 1.0:
        raise ValueError("carry_heavy_fraction must be between 0 and 1")
    if not 0.0 <= config.rehearsal_fraction <= 1.0:
        raise ValueError("rehearsal_fraction must be between 0 and 1")
    if config.rehearsal_mode not in {"padded", "layout", "uniform_layout"}:
        raise ValueError("rehearsal_mode must be 'padded', 'layout', or 'uniform_layout'")
    if not config.schedule or config.schedule[-1].digits > 1_364:
        raise ValueError("the schedule must end at or below 1,364 digits")
    if config.dims % config.num_heads:
        raise ValueError("dims must be divisible by num_heads")
    if config.dims // config.num_heads > 16:
        raise ValueError("the fused training kernel requires head_dim <= 16")
    if config.position_encoding not in {
        "none",
        "addition_column",
        "addition_column_context",
    }:
        raise ValueError(
            "position_encoding must be 'none', 'addition_column', or 'addition_column_context'"
        )
    if config.lr_decay_steps is not None and config.lr_decay_steps < config.total_steps:
        raise ValueError("lr_decay_steps must cover the complete training schedule")
    if (config.decay_half_life_min is None) != (config.decay_half_life_max is None):
        raise ValueError("both decay half-life bounds must be provided together")
    decay_half_life_range = None
    if config.decay_half_life_min is not None:
        decay_half_life_range = (
            config.decay_half_life_min,
            config.decay_half_life_max,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = KDAAdditionModel(
        vocab_size=len(TOKENS),
        dims=config.dims,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        decay_half_life_range=decay_half_life_range,
        use_fused_kernel=True,
        position_encoding=config.position_encoding,
    )
    model.train()
    optimizer = optim.AdamW(
        learning_rate=_learning_rate_schedule(config),
        weight_decay=config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())

    global_step = 0
    start_stage = 0
    start_stage_step = 0
    if resume is not None and fork_from is not None:
        raise ValueError("resume and fork_from are mutually exclusive")
    if resume is not None:
        global_step, start_stage, start_stage_step = load_checkpoint(
            resume, model, optimizer, rng, config
        )
    elif fork_from is not None:
        global_step = load_fork_checkpoint(fork_from, model, optimizer, rng)

    print(
        json.dumps(
            {
                "event": "start",
                "parameters": parameter_count(model),
                "total_steps": config.total_steps,
                "schedule": [asdict(stage) for stage in config.schedule],
                "batch_size": config.batch_size,
                "position_encoding": config.position_encoding,
                "resume_step": global_step,
                "forked_from": str(fork_from) if fork_from is not None else None,
                "output_dir": str(config.output_dir),
            }
        ),
        flush=True,
    )

    previous_widths: list[int] = []
    for stage_index, stage in enumerate(config.schedule):
        if stage_index < start_stage:
            previous_widths.append(stage.digits)
            continue
        local_start = start_stage_step if stage_index == start_stage else 0
        compiled_step = _compile_train_step(model, optimizer, config.max_grad_norm)
        window_seconds = 0.0
        window_steps = 0
        window_tokens = 0
        print(
            json.dumps(
                {
                    "event": "stage_start",
                    "stage": stage_index,
                    "digits": stage.digits,
                    "sequence_length": sequence_length(stage.digits),
                    "steps": stage.steps,
                    "starting_at": local_start,
                    "note": "the first timed step includes shape-specific compilation",
                }
            ),
            flush=True,
        )

        for stage_step in range(local_start, stage.steps):
            layout_digits = stage.digits
            active_digits = stage.digits
            if (
                config.rehearsal_mode == "uniform_layout"
                and stage.digits > 1
                and rng.random() < config.rehearsal_fraction
            ):
                # Sampling every smaller integer layout prevents the model from
                # treating a sparse set of curriculum widths as position-specific
                # tasks. The upper bound is exclusive so the non-rehearsal branch
                # alone controls the share of full-width examples.
                layout_digits = int(rng.integers(1, stage.digits))
                active_digits = layout_digits
            elif previous_widths and rng.random() < config.rehearsal_fraction:
                active_digits = int(rng.choice(previous_widths))
                if config.rehearsal_mode == "layout":
                    layout_digits = active_digits
            batch = make_addition_batch(
                rng,
                batch_size=config.batch_size,
                digits=layout_digits,
                active_digits=active_digits,
                carry_heavy_fraction=config.carry_heavy_fraction,
            )
            started = time.perf_counter()
            loss, gradient_norm = compiled_step(mx.array(batch.inputs), mx.array(batch.targets))
            mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
            duration = time.perf_counter() - started
            global_step += 1
            completed_stage_steps = stage_step + 1
            window_seconds += duration
            window_steps += 1
            window_tokens += config.batch_size * sequence_length(layout_digits)

            if global_step % config.log_every == 0 or completed_stage_steps == stage.steps:
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": global_step,
                            "stage": stage_index,
                            "stage_step": completed_stage_steps,
                            "digits": stage.digits,
                            "layout_digits": layout_digits,
                            "active_digits": active_digits,
                            "loss": float(loss),
                            "gradient_norm": float(gradient_norm),
                            "learning_rate": float(optimizer.state["learning_rate"]),
                            "step_ms": 1_000 * window_seconds / window_steps,
                            "tokens_per_second": window_tokens / window_seconds,
                        }
                    ),
                    flush=True,
                )
                window_seconds = 0.0
                window_steps = 0
                window_tokens = 0

            should_evaluate = (
                global_step % config.eval_every == 0 or completed_stage_steps == stage.steps
            )
            if should_evaluate:
                random_metrics = evaluate(
                    model,
                    rng,
                    batch_size=config.eval_batch_size,
                    digits=stage.digits,
                    carry_heavy_fraction=0.0,
                )
                carry_metrics = evaluate(
                    model,
                    rng,
                    batch_size=config.eval_batch_size,
                    digits=stage.digits,
                    carry_heavy_fraction=1.0,
                )
                print(
                    json.dumps(
                        {
                            "event": "validation",
                            "step": global_step,
                            "digits": stage.digits,
                            "random": asdict(random_metrics),
                            "carry_chain": asdict(carry_metrics),
                        }
                    ),
                    flush=True,
                )

            if global_step % config.checkpoint_every == 0 and completed_stage_steps != stage.steps:
                checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    rng,
                    config,
                    global_step=global_step,
                    stage_index=stage_index,
                    stage_step=completed_stage_steps,
                )
                print(json.dumps({"event": "checkpoint", "path": str(checkpoint)}), flush=True)

        previous_widths.append(stage.digits)
        start_stage_step = 0
        checkpoint = save_checkpoint(
            model,
            optimizer,
            rng,
            config,
            global_step=global_step,
            stage_index=stage_index + 1,
            stage_step=0,
        )
        print(json.dumps({"event": "stage_complete", "path": str(checkpoint)}), flush=True)

    print(json.dumps({"event": "complete", "step": global_step}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fused-MLX KDA model on a long-addition curriculum."
    )
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--schedule",
        help="Override the preset with DIGITS:STEPS pairs, e.g. 3:500,8:1000.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/kda-long-addition"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--fork-from",
        type=Path,
        help="Start a new curriculum while preserving a checkpoint's model and AdamW state.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--position-encoding",
        choices=("none", "addition_column", "addition_column_context"),
        default="none",
        help=(
            "Use the original NoPE path, deterministic column sinusoids, or "
            "linear-time aligned operand context for long addition."
        ),
    )
    parser.add_argument("--decay-half-life-min", type=float)
    parser.add_argument("--decay-half-life-max", type=float)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay-steps", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--carry-heavy-fraction", type=float, default=0.25)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.10)
    parser.add_argument(
        "--rehearsal-mode",
        choices=("padded", "layout", "uniform_layout"),
        default="padded",
        help=(
            "Use left-padded examples, revisit named earlier layouts, or sample "
            "every smaller integer layout uniformly."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=4_242)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    schedule = (
        parse_schedule(args.schedule)
        if args.schedule
        else (SMOKE_SCHEDULE if args.preset == "smoke" else FULL_SCHEDULE)
    )
    config = TrainConfig(
        schedule=schedule,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        dims=args.dims,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        position_encoding=args.position_encoding,
        decay_half_life_min=args.decay_half_life_min,
        decay_half_life_max=args.decay_half_life_max,
        learning_rate=args.learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        carry_heavy_fraction=args.carry_heavy_fraction,
        rehearsal_fraction=args.rehearsal_fraction,
        rehearsal_mode=args.rehearsal_mode,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    for name in ("log_every", "eval_every", "checkpoint_every"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    train(config, resume=args.resume, fork_from=args.fork_from)


if __name__ == "__main__":
    main()
