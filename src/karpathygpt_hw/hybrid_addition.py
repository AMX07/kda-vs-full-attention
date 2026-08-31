"""NoPE KDA + Gated MLA training for long, reversed-digit addition.

This module deliberately excludes both addition-column sinusoids and direct
operand-to-answer routing. The four-layer default follows the small Kimi K3
hybrid pattern: three recurrent KDA layers followed by one global Gated MLA
layer. Larger layer counts place Gated MLA at every fourth layer and at the
final layer, matching K3's global-attention placement rule.

Two training regimes are supported. The width curriculum grows operand width
stage by stage, so the operand-to-answer token offsets change at every stage.
Fixed-layout mode instead serializes every example at the full 1,364-digit
zero-padded width and samples the operand magnitude per step, keeping the
retrieval geometry stationary so the NoPE stack only has to learn one
positional code.
"""

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

from karpathygpt_hw.kda_mlx import (
    FeedForward,
    KimiDeltaAttention,
    parameter_count,
)
from karpathygpt_hw.long_addition import (
    IGNORE_INDEX,
    TOKENS,
    CurriculumStage,
    TrainConfig,
    ValidationMetrics,
    _compile_train_step,
    _learning_rate_schedule,
    evaluate,
    make_addition_batch,
    parse_schedule,
    sequence_length,
)


class GatedMultiHeadLatentAttention(nn.Module):
    """NoPE global attention with low-rank Q/KV latents and an output gate.

    This is a compact training-oriented analogue of K3's Gated MLA. Queries
    and the shared K/V source are compressed before being expanded into heads.
    The attention output is gated before its output projection.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        *,
        q_lora_rank: int = 32,
        kv_lora_rank: int = 32,
        head_dim: int | None = None,
    ):
        super().__init__()
        if dims % num_heads:
            raise ValueError("dims must be divisible by num_heads")
        if not 1 <= q_lora_rank <= dims:
            raise ValueError("q_lora_rank must be between 1 and dims")
        if not 1 <= kv_lora_rank <= dims:
            raise ValueError("kv_lora_rank must be between 1 and dims")

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = head_dim or dims // num_heads
        if self.head_dim < 1:
            raise ValueError("head_dim must be positive")
        self.attention_dims = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.q_down = nn.Linear(dims, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank, eps=1e-5)
        self.q_up = nn.Linear(q_lora_rank, self.attention_dims, bias=False)

        self.kv_down = nn.Linear(dims, kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=1e-5)
        self.k_up = nn.Linear(kv_lora_rank, self.attention_dims, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, self.attention_dims, bias=False)

        self.output_gate = nn.Linear(dims, self.attention_dims, bias=False)
        self.out_proj = nn.Linear(self.attention_dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, steps, _ = x.shape
        shape = (batch, steps, self.num_heads, self.head_dim)

        q_latent = self.q_norm(self.q_down(x))
        kv_latent = self.kv_norm(self.kv_down(x))
        q = mx.transpose(self.q_up(q_latent).reshape(shape), (0, 2, 1, 3))
        k = mx.transpose(self.k_up(kv_latent).reshape(shape), (0, 2, 1, 3))
        v = mx.transpose(self.v_up(kv_latent).reshape(shape), (0, 2, 1, 3))

        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask="causal",
        )
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(batch, steps, self.attention_dims)
        gated = attended * mx.sigmoid(self.output_gate(x))
        return self.out_proj(gated)


class HybridAttentionBlock(nn.Module):
    """Pre-norm residual block containing either KDA or Gated MLA."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        *,
        attention_type: str,
        q_lora_rank: int,
        kv_lora_rank: int,
        mla_head_dim: int,
        decay_half_life_range: tuple[float, float] | None,
    ):
        super().__init__()
        if attention_type not in {"kda", "gated_mla"}:
            raise ValueError("attention_type must be 'kda' or 'gated_mla'")
        self.attention_type = attention_type
        self.norm1 = nn.LayerNorm(dims)
        if attention_type == "kda":
            self.attention = KimiDeltaAttention(
                dims,
                num_heads,
                decay_half_life_range=decay_half_life_range,
                use_fused_kernel=True,
            )
        else:
            self.attention = GatedMultiHeadLatentAttention(
                dims,
                num_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                head_dim=mla_head_dim,
            )
        self.norm2 = nn.LayerNorm(dims)
        self.feed_forward = FeedForward(dims)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


def hybrid_layer_types(num_layers: int) -> tuple[str, ...]:
    """Return K3-style attention placement for a compact hybrid model."""

    if num_layers < 4:
        raise ValueError("the hybrid architecture requires at least four layers")
    return tuple(
        "gated_mla" if (layer % 4 == 0 or layer == num_layers) else "kda"
        for layer in range(1, num_layers + 1)
    )


class KDAHybridAdditionModel(nn.Module):
    """NoPE addition model using K3-style KDA/Gated-MLA layer placement."""

    def __init__(
        self,
        vocab_size: int = 13,
        dims: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        *,
        q_lora_rank: int = 32,
        kv_lora_rank: int = 32,
        mla_head_dim: int = 32,
        decay_half_life_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.layer_types = hybrid_layer_types(num_layers)
        # This explicit marker is saved and tested: the hybrid model is NoPE
        # and has no task-specific position or operand-routing feature.
        self.position_encoding = "none"
        self.embedding = nn.Embedding(vocab_size, dims)
        self.blocks = [
            HybridAttentionBlock(
                dims,
                num_heads,
                attention_type=attention_type,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                mla_head_dim=mla_head_dim,
                decay_half_life_range=decay_half_life_range,
            )
            for attention_type in self.layer_types
        ]
        self.final_norm = nn.LayerNorm(dims)
        self.lm_head = nn.Linear(dims, vocab_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


SMOKE_SCHEDULE = (
    CurriculumStage(1, 10),
    CurriculumStage(3, 10),
)

# The target-width stage is intentional: the deployed KDA baseline also saw
# 1,364-digit examples. Use --schedule ending below 1364 for a zero-shot length
# extrapolation experiment instead of silently changing the comparison claim.
FULL_SCHEDULE = (
    CurriculumStage(1, 2_000),
    CurriculumStage(2, 3_000),
    CurriculumStage(4, 5_000),
    CurriculumStage(8, 10_000),
    CurriculumStage(16, 20_000),
    CurriculumStage(32, 25_000),
    CurriculumStage(64, 25_000),
    CurriculumStage(128, 25_000),
    CurriculumStage(256, 25_000),
    CurriculumStage(512, 20_000),
    CurriculumStage(1_024, 10_000),
    CurriculumStage(1_364, 10_000),
)

# Fixed-layout mode: every example is serialized at the full 1,364-digit width
# with zero-padded operands, so the operand-to-answer offsets never change.
# Operand magnitude is sampled per step instead of scheduled per stage.
FIXED_SCHEDULE = (CurriculumStage(1_364, 100_000),)


@dataclass(frozen=True)
class HybridTrainConfig(TrainConfig):
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    mla_head_dim: int = 32
    adaptive_batch_size: bool = True
    max_batch_size: int = 64
    early_stop_exact_evals: int = 2
    fixed_layout: bool = False
    full_width_fraction: float = 0.25


def _stage_batch_size(config: HybridTrainConfig, digits: int) -> int:
    """Taper MLA's batch as its quadratic attention matrix grows."""

    if config.fixed_layout or not config.adaptive_batch_size:
        return config.batch_size
    if digits <= 128:
        multiplier = 16
    elif digits <= 256:
        multiplier = 8
    elif digits <= 512:
        multiplier = 4
    elif digits <= 1_024:
        multiplier = 2
    else:
        multiplier = 1
    return min(config.max_batch_size, config.batch_size * multiplier)


def _fixed_layout_probe_widths(layout_digits: int) -> tuple[int, ...]:
    return tuple(w for w in (16, 64, 512) if w < layout_digits) + (layout_digits,)


def _evaluate_fixed_layout(
    model: KDAHybridAdditionModel,
    rng: np.random.Generator,
    *,
    batch_size: int,
    layout_digits: int,
    active_digits: int,
    carry_heavy_fraction: float,
) -> ValidationMetrics:
    """Teacher-forced metrics for an active width inside the fixed layout."""

    batch = make_addition_batch(
        rng,
        batch_size=batch_size,
        digits=layout_digits,
        active_digits=active_digits,
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
        answer_exact_accuracy=float(
            np.mean(np.all(predicted_answer == expected_answer, axis=1))
        ),
        trailing_space_accuracy=float(np.mean(predicted[:, -1] == batch.targets[:, -1])),
    )
    model.train()
    return metrics


def _architecture_metadata(config: HybridTrainConfig) -> dict[str, Any]:
    return {
        "name": "kda_gated_mla_nope",
        "layer_types": list(hybrid_layer_types(config.num_layers)),
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "mla_head_dim": config.mla_head_dim,
        "position_encoding": "none",
        "arithmetic_column_sinusoid": False,
        "aligned_operand_context": False,
        "reversed_answer_digits": True,
    }


def _config_metadata(config: HybridTrainConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["schedule"] = [asdict(stage) for stage in config.schedule]
    metadata["output_dir"] = str(config.output_dir)
    return metadata


def _save_checkpoint(
    model: KDAHybridAdditionModel,
    optimizer: optim.Optimizer,
    rng: np.random.Generator,
    config: HybridTrainConfig,
    *,
    global_step: int,
    stage_index: int,
    stage_step: int,
) -> Path:
    checkpoint_dir = config.output_dir / f"checkpoint-{global_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(model.parameters(), optimizer.state)

    model_temp = checkpoint_dir / "model.tmp.safetensors"
    optimizer_temp = checkpoint_dir / "optimizer.tmp.safetensors"
    state_temp = checkpoint_dir / "state.tmp.json"
    model.save_weights(str(model_temp))
    mx.save_safetensors(optimizer_temp, dict(tree_flatten(optimizer.state)))
    state_temp.write_text(
        json.dumps(
            {
                "global_step": global_step,
                "stage_index": stage_index,
                "stage_step": stage_step,
                "rng_state": rng.bit_generator.state,
                "architecture": _architecture_metadata(config),
                "config": _config_metadata(config),
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(model_temp, checkpoint_dir / "model.safetensors")
    os.replace(optimizer_temp, checkpoint_dir / "optimizer.safetensors")
    os.replace(state_temp, checkpoint_dir / "state.json")

    latest_temp = config.output_dir / "latest.tmp.json"
    latest_temp.write_text(json.dumps({"checkpoint": checkpoint_dir.name}) + "\n")
    os.replace(latest_temp, config.output_dir / "latest.json")
    return checkpoint_dir


def _load_checkpoint(
    checkpoint_dir: Path,
    model: KDAHybridAdditionModel,
    optimizer: optim.Optimizer,
    rng: np.random.Generator,
    config: HybridTrainConfig,
    *,
    exact_resume: bool,
) -> tuple[int, int, int]:
    state = json.loads((checkpoint_dir / "state.json").read_text())
    expected_architecture = _architecture_metadata(config)
    if state.get("architecture") != expected_architecture:
        raise ValueError("checkpoint architecture does not match the NoPE KDA/Gated-MLA model")
    if exact_resume:
        saved_config = state["config"]
        current_config = _config_metadata(config)
        # Defaults let checkpoints written before a field existed still resume.
        legacy_defaults = {"fixed_layout": False, "full_width_fraction": 0.25}
        for key in (
            "schedule",
            "batch_size",
            "adaptive_batch_size",
            "max_batch_size",
            "dims",
            "num_heads",
            "num_layers",
            "q_lora_rank",
            "kv_lora_rank",
            "mla_head_dim",
            "decay_half_life_min",
            "decay_half_life_max",
            "lr_decay_steps",
            "rehearsal_mode",
            "fixed_layout",
            "full_width_fraction",
        ):
            saved_value = saved_config.get(key, legacy_defaults.get(key))
            if saved_value != current_config[key]:
                raise ValueError(f"resume configuration mismatch for {key}")

    model.load_weights(str(checkpoint_dir / "model.safetensors"))
    optimizer_arrays = mx.load(str(checkpoint_dir / "optimizer.safetensors"))
    optimizer.state = tree_unflatten(list(optimizer_arrays.items()))
    rng.bit_generator.state = state["rng_state"]
    mx.eval(model.parameters(), optimizer.state)
    stage_index = int(state["stage_index"]) if exact_resume else 0
    stage_step = int(state["stage_step"]) if exact_resume else 0
    return int(state["global_step"]), stage_index, stage_step


def _save_best_pointer(
    config: HybridTrainConfig,
    checkpoint: Path,
    *,
    digits: int,
    global_step: int,
    random_exact: float,
    carry_exact: float,
) -> None:
    payload = {
        "checkpoint": checkpoint.name,
        "digits": digits,
        "step": global_step,
        "random_exact_accuracy": random_exact,
        "carry_exact_accuracy": carry_exact,
    }
    for filename in ("best.json", f"best-{digits:04d}.json"):
        temporary = config.output_dir / f"{filename}.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, config.output_dir / filename)


def _validate_config(config: HybridTrainConfig) -> tuple[float, float] | None:
    if not mx.metal.is_available():
        raise RuntimeError("hybrid long-addition training requires Apple Metal")
    if config.position_encoding != "none":
        raise ValueError("the KDA/Gated-MLA comparison is NoPE-only")
    if config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if config.max_batch_size < config.batch_size:
        raise ValueError("max_batch_size cannot be smaller than batch_size")
    if not config.schedule or config.schedule[-1].digits > 1_364:
        raise ValueError("the schedule must end at or below 1,364 digits")
    if config.fixed_layout and len(config.schedule) != 1:
        raise ValueError("fixed-layout training uses a single-stage schedule")
    if not 0.0 <= config.full_width_fraction <= 1.0:
        raise ValueError("full_width_fraction must be between 0 and 1")
    if config.dims % config.num_heads:
        raise ValueError("dims must be divisible by num_heads")
    if config.dims // config.num_heads > 16:
        raise ValueError("the fused KDA training kernel requires head_dim <= 16")
    hybrid_layer_types(config.num_layers)
    if not 1 <= config.q_lora_rank <= config.dims:
        raise ValueError("q_lora_rank must be between 1 and dims")
    if not 1 <= config.kv_lora_rank <= config.dims:
        raise ValueError("kv_lora_rank must be between 1 and dims")
    if config.mla_head_dim < 1:
        raise ValueError("mla_head_dim must be positive")
    if config.early_stop_exact_evals < 0:
        raise ValueError("early_stop_exact_evals cannot be negative")
    if not 0.0 <= config.carry_heavy_fraction <= 1.0:
        raise ValueError("carry_heavy_fraction must be between 0 and 1")
    if not 0.0 <= config.rehearsal_fraction <= 1.0:
        raise ValueError("rehearsal_fraction must be between 0 and 1")
    if config.rehearsal_mode not in {"padded", "layout", "uniform_layout"}:
        raise ValueError("invalid rehearsal_mode")
    if config.lr_decay_steps is not None and config.lr_decay_steps < config.total_steps:
        raise ValueError("lr_decay_steps must cover the complete training schedule")
    if (config.decay_half_life_min is None) != (config.decay_half_life_max is None):
        raise ValueError("both decay half-life bounds must be provided together")
    if config.decay_half_life_min is None:
        return None
    return config.decay_half_life_min, config.decay_half_life_max


class _RunLogger:
    """Mirror structured events to stdout and the run's durable JSONL log."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "training.jsonl"

    def __call__(self, event: dict[str, Any]) -> None:
        line = json.dumps(event)
        print(line, flush=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")


def train(
    config: HybridTrainConfig,
    *,
    resume: Path | None = None,
    fork_from: Path | None = None,
) -> None:
    """Train the NoPE hybrid while preserving apples-to-apples logging."""

    decay_half_life_range = _validate_config(config)
    if resume is not None and fork_from is not None:
        raise ValueError("resume and fork_from are mutually exclusive")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_event = _RunLogger(config.output_dir)
    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = KDAHybridAdditionModel(
        vocab_size=len(TOKENS),
        dims=config.dims,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        mla_head_dim=config.mla_head_dim,
        decay_half_life_range=decay_half_life_range,
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
    if resume is not None:
        global_step, start_stage, start_stage_step = _load_checkpoint(
            resume,
            model,
            optimizer,
            rng,
            config,
            exact_resume=True,
        )
    elif fork_from is not None:
        global_step, _, _ = _load_checkpoint(
            fork_from,
            model,
            optimizer,
            rng,
            config,
            exact_resume=False,
        )

    log_event(
        {
            "event": "start",
            "architecture": _architecture_metadata(config),
            "parameters": parameter_count(model),
            "total_steps": config.total_steps,
            "schedule": [asdict(stage) for stage in config.schedule],
            "long_context_batch_size": config.batch_size,
            "adaptive_batch_size": config.adaptive_batch_size,
            "max_batch_size": config.max_batch_size,
            "fixed_layout": config.fixed_layout,
            "full_width_fraction": config.full_width_fraction,
            "resume_step": global_step,
            "forked_from": str(fork_from) if fork_from is not None else None,
            "output_dir": str(config.output_dir),
        }
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
        best_stage_score = -1.0
        perfect_validation_streak = 0
        stop_stage_early = False
        stage_batch_size = _stage_batch_size(config, stage.digits)
        log_event(
            {
                "event": "stage_start",
                "stage": stage_index,
                "digits": stage.digits,
                "sequence_length": sequence_length(stage.digits),
                "batch_size": stage_batch_size,
                "steps": stage.steps,
                "starting_at": local_start,
                "note": "the first timed step includes shape-specific compilation",
            }
        )

        for stage_step in range(local_start, stage.steps):
            layout_digits = stage.digits
            active_digits = stage.digits
            if config.fixed_layout:
                if stage.digits > 1 and rng.random() >= config.full_width_fraction:
                    active_digits = int(rng.integers(1, stage.digits))
            elif (
                config.rehearsal_mode == "uniform_layout"
                and stage.digits > 1
                and rng.random() < config.rehearsal_fraction
            ):
                layout_digits = int(rng.integers(1, stage.digits))
                active_digits = layout_digits
            elif previous_widths and rng.random() < config.rehearsal_fraction:
                active_digits = int(rng.choice(previous_widths))
                if config.rehearsal_mode == "layout":
                    layout_digits = active_digits
            batch = make_addition_batch(
                rng,
                batch_size=stage_batch_size,
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
            window_tokens += stage_batch_size * sequence_length(layout_digits)

            if global_step % config.log_every == 0 or completed_stage_steps == stage.steps:
                log_event(
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
                )
                window_seconds = 0.0
                window_steps = 0
                window_tokens = 0

            should_evaluate = (
                global_step % config.eval_every == 0 or completed_stage_steps == stage.steps
            )
            if should_evaluate:
                if config.fixed_layout:
                    probe_metrics: dict[str, Any] = {}
                    worst_exact = 1.0
                    for probe in _fixed_layout_probe_widths(stage.digits):
                        probe_random = _evaluate_fixed_layout(
                            model,
                            rng,
                            batch_size=config.eval_batch_size,
                            layout_digits=stage.digits,
                            active_digits=probe,
                            carry_heavy_fraction=0.0,
                        )
                        probe_carry = _evaluate_fixed_layout(
                            model,
                            rng,
                            batch_size=config.eval_batch_size,
                            layout_digits=stage.digits,
                            active_digits=probe,
                            carry_heavy_fraction=1.0,
                        )
                        probe_metrics[str(probe)] = {
                            "random": asdict(probe_random),
                            "carry_chain": asdict(probe_carry),
                        }
                        worst_exact = min(
                            worst_exact,
                            probe_random.answer_exact_accuracy,
                            probe_carry.answer_exact_accuracy,
                        )
                        if probe == stage.digits:
                            random_metrics = probe_random
                            carry_metrics = probe_carry
                    log_event(
                        {
                            "event": "validation",
                            "step": global_step,
                            "digits": stage.digits,
                            "fixed_layout": True,
                            "probes": probe_metrics,
                        }
                    )
                    random_exact = random_metrics.answer_exact_accuracy
                    carry_exact = carry_metrics.answer_exact_accuracy
                    # Digit accuracy breaks ties so "best" keeps advancing
                    # during the long span before any probe is exactly right.
                    stage_score = worst_exact + 1e-3 * (
                        random_metrics.answer_digit_accuracy
                        + carry_metrics.answer_digit_accuracy
                    )
                else:
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
                    log_event(
                        {
                            "event": "validation",
                            "step": global_step,
                            "digits": stage.digits,
                            "random": asdict(random_metrics),
                            "carry_chain": asdict(carry_metrics),
                        }
                    )
                    random_exact = random_metrics.answer_exact_accuracy
                    carry_exact = carry_metrics.answer_exact_accuracy
                    worst_exact = min(random_exact, carry_exact)
                    stage_score = worst_exact
                if stage_score > best_stage_score:
                    best_stage_score = stage_score
                    best_checkpoint = _save_checkpoint(
                        model,
                        optimizer,
                        rng,
                        config,
                        global_step=global_step,
                        stage_index=stage_index,
                        stage_step=completed_stage_steps,
                    )
                    _save_best_pointer(
                        config,
                        best_checkpoint,
                        digits=stage.digits,
                        global_step=global_step,
                        random_exact=random_exact,
                        carry_exact=carry_exact,
                    )
                    log_event(
                        {
                            "event": "best_checkpoint",
                            "path": str(best_checkpoint),
                            "digits": stage.digits,
                            "score": stage_score,
                        }
                    )
                if worst_exact == 1.0:
                    perfect_validation_streak += 1
                else:
                    perfect_validation_streak = 0
                if (
                    config.early_stop_exact_evals
                    and perfect_validation_streak >= config.early_stop_exact_evals
                ):
                    stop_stage_early = True
                    log_event(
                        {
                            "event": "stage_early_stop",
                            "digits": stage.digits,
                            "step": global_step,
                            "consecutive_perfect_evals": perfect_validation_streak,
                        }
                    )

            if global_step % config.checkpoint_every == 0 and completed_stage_steps != stage.steps:
                checkpoint = _save_checkpoint(
                    model,
                    optimizer,
                    rng,
                    config,
                    global_step=global_step,
                    stage_index=stage_index,
                    stage_step=completed_stage_steps,
                )
                log_event({"event": "checkpoint", "path": str(checkpoint)})

            if stop_stage_early:
                break

        previous_widths.append(stage.digits)
        start_stage_step = 0
        checkpoint = _save_checkpoint(
            model,
            optimizer,
            rng,
            config,
            global_step=global_step,
            stage_index=stage_index + 1,
            stage_step=0,
        )
        log_event({"event": "stage_complete", "path": str(checkpoint)})

    log_event({"event": "complete", "step": global_step})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a NoPE 3xKDA + 1xGated-MLA model on reversed-digit long addition. "
            "No arithmetic-column sinusoid or aligned operand context is available."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "full", "fixed"),
        default="smoke",
        help=(
            "'fixed' trains at a single 1,364-digit zero-padded layout with "
            "per-step random operand widths instead of a width curriculum."
        ),
    )
    parser.add_argument("--schedule", help="Override with DIGITS:STEPS pairs.")
    parser.add_argument(
        "--fixed-layout",
        action="store_true",
        help="Force fixed-layout sampling with a custom single-stage --schedule.",
    )
    parser.add_argument(
        "--full-width-fraction",
        type=float,
        default=0.25,
        help=(
            "Fixed layout only: fraction of steps trained at the full operand "
            "width; the rest sample a uniform width in [1, layout)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/kda-gated-mla-long-addition"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fork-from", type=Path)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size at 1,364 digits (and at every width with --fixed-batch-size).",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument(
        "--fixed-batch-size",
        action="store_true",
        help="Disable the default short-context batch-size scaling.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--q-lora-rank", type=int, default=32)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--mla-head-dim", type=int, default=32)
    parser.add_argument("--decay-half-life-min", type=float)
    parser.add_argument("--decay-half-life-max", type=float)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=250_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--carry-heavy-fraction", type=float, default=0.5)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.0)
    parser.add_argument(
        "--rehearsal-mode",
        choices=("padded", "layout", "uniform_layout"),
        default="uniform_layout",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument(
        "--early-stop-exact-evals",
        type=int,
        default=2,
        help="Advance after this many consecutive perfect random/carry validations; 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=4_242)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preset_schedules = {
        "smoke": SMOKE_SCHEDULE,
        "full": FULL_SCHEDULE,
        "fixed": FIXED_SCHEDULE,
    }
    schedule = (
        parse_schedule(args.schedule)
        if args.schedule
        else preset_schedules[args.preset]
    )
    fixed_layout = args.fixed_layout or args.preset == "fixed"
    config = HybridTrainConfig(
        schedule=schedule,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        dims=args.dims,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        position_encoding="none",
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
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        mla_head_dim=args.mla_head_dim,
        adaptive_batch_size=not args.fixed_batch_size,
        max_batch_size=args.max_batch_size,
        early_stop_exact_evals=args.early_stop_exact_evals,
        fixed_layout=fixed_layout,
        full_width_fraction=args.full_width_fraction,
    )
    for name in ("log_every", "eval_every", "checkpoint_every"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    train(config, resume=args.resume, fork_from=args.fork_from)


if __name__ == "__main__":
    main()
