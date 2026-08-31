"""Variable-length unpadded reversed addition with a dense MLX Transformer.

This is the full-attention arm of the architecture comparison, ported from the
PyTorch module :mod:`karpathygpt_hw.variable_addition` to MLX so that it shares
a framework with the hybrid arm.  Only then are wall-clock and memory numbers a
statement about attention architecture rather than about MLX versus PyTorch.

The model is the notebook architecture unchanged: learned absolute position
embeddings, pre-norm blocks, dense causal multi-head self-attention, and a
four-times-wide ReLU feed-forward.  Data sampling, evaluation, logging, and the
optimizer are imported from the hybrid module, so the two arms differ only in
the model class.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from karpathygpt_hw.kda_mlx import parameter_count
from karpathygpt_hw.variable_addition import (
    TOKENS,
    build_eval_groups,
    valid_layouts,
)
from karpathygpt_hw.variable_hybrid_addition import (
    _append_event,
    _compile_train_step,
    _write_json,
    evaluate_groups,
    sample_training_batch,
)


@dataclass(frozen=True)
class DenseVariableConfig:
    max_digits: int
    output_dir: Path
    steps: int = 5_000
    batch_size: int = 256
    dims: int = 64
    num_heads: int = 4
    num_layers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_carry_fraction: float = 0.25
    eval_every: int = 250
    log_every: int = 50
    quick_eval_cases: int = 128
    final_eval_cases: int = 256
    seed: int = 4_242
    early_stop_evals: int = 3
    early_stop_threshold: float = 1.0
    attention_impl: str = "fused"

    @property
    def block_size(self) -> int:
        return 3 * self.max_digits + 3


class DenseCausalSelfAttention(nn.Module):
    """Dense causal multi-head attention matching the PyTorch reference."""

    def __init__(self, dims: int, num_heads: int, *, impl: str = "fused"):
        super().__init__()
        if dims % num_heads:
            raise ValueError("dims must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dims = dims // num_heads
        self.scale = self.head_dims**-0.5
        self.impl = impl
        self.key = nn.Linear(dims, dims, bias=False)
        self.query = nn.Linear(dims, dims, bias=False)
        self.value = nn.Linear(dims, dims, bias=False)
        self.projection = nn.Linear(dims, dims)

    def __call__(self, inputs: mx.array) -> mx.array:
        batch, length, dims = inputs.shape
        shape = (batch, length, self.num_heads, self.head_dims)
        queries = mx.transpose(self.query(inputs).reshape(shape), (0, 2, 1, 3))
        keys = mx.transpose(self.key(inputs).reshape(shape), (0, 2, 1, 3))
        values = mx.transpose(self.value(inputs).reshape(shape), (0, 2, 1, 3))

        if self.impl == "fused":
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask="causal"
            )
        else:
            # Explicit matmul/mask/softmax, mirroring the PyTorch reference so
            # a cross-framework timing comparison is like-for-like.
            scores = (queries @ mx.transpose(keys, (0, 1, 3, 2))) * self.scale
            causal = mx.tril(mx.ones((length, length), dtype=mx.bool_))
            scores = mx.where(causal, scores, mx.array(-mx.inf, scores.dtype))
            output = mx.softmax(scores, axis=-1) @ values

        output = mx.transpose(output, (0, 2, 1, 3)).reshape(batch, length, dims)
        return self.projection(output)


class FeedForward(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.up = nn.Linear(dims, 4 * dims)
        self.down = nn.Linear(4 * dims, dims)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down(nn.relu(self.up(inputs)))


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, *, impl: str):
        super().__init__()
        self.attention = DenseCausalSelfAttention(dims, num_heads, impl=impl)
        self.feed_forward = FeedForward(dims)
        self.norm_attention = nn.LayerNorm(dims)
        self.norm_feed_forward = nn.LayerNorm(dims)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = inputs + self.attention(self.norm_attention(inputs))
        return hidden + self.feed_forward(self.norm_feed_forward(hidden))


class MLXAdditionTransformer(nn.Module):
    """Learned-absolute-position dense Transformer, MLX port of the notebook model."""

    def __init__(
        self,
        *,
        block_size: int,
        dims: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        attention_impl: str = "fused",
    ):
        super().__init__()
        self.block_size = block_size
        self.position_encoding = "learned_absolute"
        self.token_embeddings = nn.Embedding(len(TOKENS), dims)
        self.position_embeddings = nn.Embedding(block_size, dims)
        self.blocks = [
            TransformerBlock(dims, num_heads, impl=attention_impl) for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(dims)
        self.lm_head = nn.Linear(dims, len(TOKENS))
        # PyTorch's nn.Embedding initializes from N(0, 1) whereas MLX defaults to
        # roughly 1/sqrt(dims). Match the reference so the two framework arms
        # differ only in framework, not in starting distribution.
        self.token_embeddings.weight = mx.random.normal(self.token_embeddings.weight.shape)
        self.position_embeddings.weight = mx.random.normal(self.position_embeddings.weight.shape)

    def __call__(self, token_ids: mx.array) -> mx.array:
        length = token_ids.shape[1]
        if length > self.block_size:
            raise ValueError(f"input length {length} exceeds block size {self.block_size}")
        hidden = self.token_embeddings(token_ids) + self.position_embeddings(mx.arange(length))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def _jsonable_config(config: DenseVariableConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _validate_config(config: DenseVariableConfig) -> None:
    if config.max_digits < 1:
        raise ValueError("max_digits must be positive")
    if config.batch_size < 1 or config.steps < 1:
        raise ValueError("batch_size and steps must be positive")
    if config.dims % config.num_heads:
        raise ValueError("dims must be divisible by num_heads")
    if config.attention_impl not in {"fused", "manual"}:
        raise ValueError("attention_impl must be 'fused' or 'manual'")
    if not 0.0 <= config.hard_carry_fraction <= 1.0:
        raise ValueError("hard_carry_fraction must be between 0 and 1")


def train(config: DenseVariableConfig) -> dict[str, Any]:
    """Train and fully evaluate one dense-attention MLX experiment."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    _write_json(config.output_dir / "config.json", _jsonable_config(config))

    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    layouts = valid_layouts(config.max_digits)
    model = MLXAdditionTransformer(
        block_size=config.block_size,
        dims=config.dims,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        attention_impl=config.attention_impl,
    )
    model.train()
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    compiled_step = _compile_train_step(model, optimizer, config.max_grad_norm)
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
        "data": "variable_length_unpadded_reversed_answer",
        "parameters": parameter_count(model),
        "device": str(mx.default_device()),
        "platform": platform.platform(),
        "layout_count": len(layouts),
        "framework": "mlx",
    }
    _append_event(metrics_path, metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)

    mx.eval(model.parameters(), optimizer.state)
    mx.synchronize()
    mx.reset_peak_memory()
    train_started = time.perf_counter()
    interval_started = train_started
    interval_examples = 0
    interval_tokens = 0
    last_loss = float("nan")
    last_gradient_norm = float("nan")
    perfect_streak = 0
    converged_step: int | None = None
    stopped_early = False

    for step in range(1, config.steps + 1):
        batch = sample_training_batch(rng, layouts, config)
        loss, gradient_norm = compiled_step(mx.array(batch.inputs), mx.array(batch.targets))
        mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
        last_loss = float(loss)
        last_gradient_norm = float(gradient_norm)
        interval_examples += len(batch.examples)
        interval_tokens += batch.inputs.size

        if step % config.log_every == 0 or step == 1:
            mx.synchronize()
            now = time.perf_counter()
            elapsed = now - interval_started
            event = {
                "event": "train",
                "step": step,
                "loss": last_loss,
                "gradient_norm": last_gradient_norm,
                "layout": asdict(batch.layout),
                "examples_per_second": interval_examples / elapsed,
                "tokens_per_second": interval_tokens / elapsed,
                "elapsed_training_seconds": now - train_started,
                "active_memory_bytes": int(mx.get_active_memory()),
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
            _append_event(metrics_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            interval_started = now
            interval_examples = 0
            interval_tokens = 0

        if step % config.eval_every == 0 or step == config.steps:
            mx.synchronize()
            eval_started = time.perf_counter()
            validation = evaluate_groups(model, quick_groups)
            mx.synchronize()
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
                    {key: value for key, value in event.items() if key != "groups"},
                    sort_keys=True,
                ),
                flush=True,
            )

            # Stop once the quick evaluation has held at the threshold for
            # several consecutive checks. Evaluation is noisy right after
            # convergence, so a streak is required and the reported convergence
            # step is the first evaluation of the streak, not the last.
            if validation["micro_answer_exact_accuracy"] >= config.early_stop_threshold:
                perfect_streak += 1
                if converged_step is None:
                    converged_step = step
            else:
                perfect_streak = 0
                converged_step = None
            if config.early_stop_evals and perfect_streak >= config.early_stop_evals:
                stopped_early = True
                stop_event = {
                    "event": "early_stop",
                    "step": step,
                    "converged_step": converged_step,
                    "consecutive_evals": perfect_streak,
                }
                _append_event(metrics_path, stop_event)
                print(json.dumps(stop_event, sort_keys=True), flush=True)
                break

    mx.synchronize()
    training_seconds = time.perf_counter() - train_started
    model.save_weights(str(config.output_dir / "model.safetensors"))
    mx.save_safetensors(
        str(config.output_dir / "optimizer.safetensors"),
        dict(tree_flatten(optimizer.state)),
    )

    final_groups = build_eval_groups(
        max_digits=config.max_digits,
        cases_per_layout=config.final_eval_cases,
        split="test",
        seed=config.seed,
        quick=False,
    )
    mx.synchronize()
    final_eval_started = time.perf_counter()
    final_metrics = evaluate_groups(model, final_groups)
    mx.synchronize()
    final_eval_seconds = time.perf_counter() - final_eval_started
    result = {
        "event": "complete",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_digits": config.max_digits,
        "parameters": parameter_count(model),
        "steps": config.steps,
        "completed_steps": step,
        "converged_step": converged_step,
        "stopped_early": stopped_early,
        "final_training_loss": last_loss,
        "final_gradient_norm": last_gradient_norm,
        "training_seconds": training_seconds,
        "final_evaluation_seconds": final_eval_seconds,
        "total_seconds": training_seconds + final_eval_seconds,
        "peak_metal_memory_bytes": int(mx.get_peak_memory()),
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
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hard-carry-fraction", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--quick-eval-cases", type=int, default=128)
    parser.add_argument("--final-eval-cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4_242)
    parser.add_argument(
        "--early-stop-evals",
        type=int,
        default=3,
        help="Stop after this many consecutive quick evals at the threshold; 0 disables.",
    )
    parser.add_argument("--early-stop-threshold", type=float, default=1.0)
    parser.add_argument("--attention-impl", choices=("fused", "manual"), default="fused")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(DenseVariableConfig(**vars(args)))


if __name__ == "__main__":
    main()
