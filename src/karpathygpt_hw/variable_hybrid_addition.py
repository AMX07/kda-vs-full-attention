"""Variable-length, unpadded, reversed addition with K3-style hybrid attention.

The data protocol is shared with :mod:`karpathygpt_hw.variable_addition`, but
the learned-position dense Transformer is replaced by the project's small NoPE
hybrid: three fused KDA layers followed by one gated MLA layer.  This is an
attention-level analogue of Kimi K3, not a reproduction of K3's MoE, AttnRes,
multimodal, or quantized architecture.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from karpathygpt_hw.hybrid_addition import KDAHybridAdditionModel
from karpathygpt_hw.kda_mlx import masked_addition_loss, parameter_count
from karpathygpt_hw.variable_addition import (
    EOS_TOKEN,
    IGNORE_INDEX,
    TOKENS,
    AdditionExample,
    AdditionLayout,
    EvalGroup,
    _sample_full_carry_pair,
    _sample_layout_pair,
    build_eval_groups,
    carry_chain_length,
    decode,
    encode,
    example_layout,
    format_answer,
    format_row,
    valid_layouts,
)


@dataclass(frozen=True)
class HybridVariableConfig:
    max_digits: int
    output_dir: Path
    steps: int = 5_000
    batch_size: int = 256
    dims: int = 64
    num_heads: int = 4
    num_layers: int = 4
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    mla_head_dim: int = 32
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

    @property
    def block_size(self) -> int:
        """Largest shifted sequence used, retained for dense-run comparison."""

        return 3 * self.max_digits + 3


@dataclass(frozen=True)
class MLXAdditionBatch:
    inputs: np.ndarray
    targets: np.ndarray
    layout: AdditionLayout
    examples: tuple[AdditionExample, ...]


def make_mlx_batch(examples: Iterable[AdditionExample]) -> MLXAdditionBatch:
    """Create an unpadded MLX-ready batch for one exact decimal layout."""

    examples_tuple = tuple(examples)
    if not examples_tuple:
        raise ValueError("cannot make an empty batch")
    layout = example_layout(examples_tuple[0])
    if any(example_layout(example) != layout for example in examples_tuple):
        raise ValueError("all examples in an unpadded batch must share one exact layout")

    rows = [encode(format_row(example, "reversed")) for example in examples_tuple]
    if any(len(row) != layout.row_length for row in rows):
        raise AssertionError("layout and encoded row length disagree")
    full = np.asarray(rows, dtype=np.int32)
    inputs = full[:, :-1]
    targets = full[:, 1:].copy()
    targets[:, : layout.prompt_length - 1] = IGNORE_INDEX
    return MLXAdditionBatch(inputs, targets, layout, examples_tuple)


def sample_training_batch(
    rng: np.random.Generator,
    layouts: tuple[AdditionLayout, ...],
    config: HybridVariableConfig,
) -> MLXAdditionBatch:
    """Use the same layout-balanced and explicit-carry sampler as the dense run."""

    if rng.random() < config.hard_carry_fraction:
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
    return make_mlx_batch(examples)


def _compile_train_step(
    model: KDAHybridAdditionModel,
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


def evaluate_group(model: KDAHybridAdditionModel, group: EvalGroup) -> dict[str, Any]:
    """Teacher-force loss, then greedily generate the answer from only ``A+B=``."""

    batch = make_mlx_batch(group.examples)
    inputs = mx.array(batch.inputs)
    targets = mx.array(batch.targets)
    loss = masked_addition_loss(model, inputs, targets)

    prompts = [f"{example.a}+{example.b}=" for example in group.examples]
    generated = mx.array(np.asarray([encode(prompt) for prompt in prompts], dtype=np.int32))
    for _ in range(group.layout.digits_answer + 1):
        next_token = mx.argmax(model(generated)[:, -1, :], axis=-1, keepdims=True)
        generated = mx.concatenate((generated, next_token), axis=1)

    continuation = generated[:, group.layout.prompt_length :]
    expected_text = [
        format_answer(example, "reversed") + EOS_TOKEN for example in group.examples
    ]
    expected = np.asarray([encode(text) for text in expected_text], dtype=np.int32)
    mx.eval(loss, continuation)
    predicted = np.asarray(continuation)
    exact = np.all(predicted == expected, axis=1)
    answer_exact = np.all(
        predicted[:, : group.layout.digits_answer]
        == expected[:, : group.layout.digits_answer],
        axis=1,
    )
    digit_correct = (
        predicted[:, : group.layout.digits_answer]
        == expected[:, : group.layout.digits_answer]
    )
    eos_correct = predicted[:, -1] == expected[:, -1]
    predictions = [decode(row.tolist()) for row in predicted]
    return {
        "name": group.name,
        "layout": asdict(group.layout),
        "cases": len(group.examples),
        "loss": float(loss),
        "sequence_exact_accuracy": float(np.mean(exact)),
        "answer_exact_accuracy": float(np.mean(answer_exact)),
        "answer_digit_accuracy": float(np.mean(digit_correct)),
        "eos_accuracy": float(np.mean(eos_correct)),
        "correct_answers": int(np.sum(answer_exact)),
        "correct_digits": int(np.sum(digit_correct)),
        "answer_digits": int(digit_correct.size),
        "examples": [
            {
                "a": example.a,
                "b": example.b,
                "expected_answer": expected_value.rstrip(EOS_TOKEN),
                "predicted_answer": prediction.rstrip(EOS_TOKEN),
                "correct": (
                    prediction.rstrip(EOS_TOKEN) == expected_value.rstrip(EOS_TOKEN)
                ),
                "carry_chain": carry_chain_length(example.a, example.b),
            }
            for example, expected_value, prediction in zip(
                group.examples[:8], expected_text[:8], predictions[:8], strict=True
            )
        ],
    }


def evaluate_groups(
    model: KDAHybridAdditionModel,
    groups: tuple[EvalGroup, ...],
) -> dict[str, Any]:
    model.eval()
    group_metrics = [evaluate_group(model, group) for group in groups]
    total_cases = sum(metric["cases"] for metric in group_metrics)
    total_correct = sum(metric["correct_answers"] for metric in group_metrics)
    total_digits = sum(metric["answer_digits"] for metric in group_metrics)
    total_correct_digits = sum(metric["correct_digits"] for metric in group_metrics)
    layout_metrics = [metric for metric in group_metrics if metric["name"].startswith("layout-")]
    hard_metrics = [
        metric for metric in group_metrics if metric["name"].startswith("hard-carry-")
    ]
    result = {
        "cases": total_cases,
        "micro_answer_exact_accuracy": total_correct / total_cases if total_cases else 0.0,
        "micro_answer_digit_accuracy": (
            total_correct_digits / total_digits if total_digits else 0.0
        ),
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


def _jsonable_config(config: HybridVariableConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    metadata["block_size"] = config.block_size
    return metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def _validate_config(config: HybridVariableConfig) -> None:
    if not mx.metal.is_available():
        raise RuntimeError("hybrid variable-addition training requires Apple Metal")
    if config.max_digits < 1:
        raise ValueError("max_digits must be positive")
    if config.batch_size < 1 or config.steps < 1:
        raise ValueError("batch size and steps must be positive")
    if config.dims % config.num_heads:
        raise ValueError("dims must be divisible by num_heads")
    if config.dims // config.num_heads > 16:
        raise ValueError("the fused KDA training kernel requires head_dim <= 16")
    if config.num_layers < 4:
        raise ValueError("K3-style hybrid placement requires at least four layers")
    if not 1 <= config.q_lora_rank <= config.dims:
        raise ValueError("q_lora_rank must be between 1 and dims")
    if not 1 <= config.kv_lora_rank <= config.dims:
        raise ValueError("kv_lora_rank must be between 1 and dims")
    if config.mla_head_dim < 1:
        raise ValueError("mla_head_dim must be positive")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if not 0 <= config.hard_carry_fraction <= 1:
        raise ValueError("hard_carry_fraction must be between zero and one")
    if config.eval_every < 1 or config.log_every < 1:
        raise ValueError("logging intervals must be positive")


def train(config: HybridVariableConfig) -> dict[str, Any]:
    """Train and fully evaluate one fresh maximum-digit experiment."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    _write_json(config.output_dir / "config.json", _jsonable_config(config))

    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    layouts = valid_layouts(config.max_digits)
    model = KDAHybridAdditionModel(
        vocab_size=len(TOKENS),
        dims=config.dims,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        mla_head_dim=config.mla_head_dim,
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
        "architecture": "k3_style_3kda_1gated_mla_nope_dense_ffn_analogue",
        "layer_types": list(model.layer_types),
        "data": "variable_length_unpadded_reversed_answer",
        "parameters": parameter_count(model),
        "device": str(mx.default_device()),
        "platform": platform.platform(),
        "layout_count": len(layouts),
        "kda_training_path": "fused_metal_custom_vjp",
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
    parser.add_argument("--q-lora-rank", type=int, default=32)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--mla-head-dim", type=int, default=32)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(HybridVariableConfig(**vars(args)))


if __name__ == "__main__":
    main()
