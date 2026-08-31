import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from karpathygpt_hw.hybrid_addition import (  # noqa: E402
    GatedMultiHeadLatentAttention,
    HybridTrainConfig,
    KDAHybridAdditionModel,
    _architecture_metadata,
    _stage_batch_size,
    hybrid_layer_types,
    train,
)
from karpathygpt_hw.kda_mlx import masked_addition_loss, parameter_count  # noqa: E402
from karpathygpt_hw.long_addition import parse_schedule  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="hybrid training uses the custom Metal KDA VJP",
)


def test_hybrid_uses_k3_style_layer_placement() -> None:
    assert hybrid_layer_types(4) == ("kda", "kda", "kda", "gated_mla")
    assert hybrid_layer_types(8) == (
        "kda",
        "kda",
        "kda",
        "gated_mla",
        "kda",
        "kda",
        "kda",
        "gated_mla",
    )
    assert hybrid_layer_types(5)[-2:] == ("gated_mla", "gated_mla")
    with pytest.raises(ValueError, match="at least four"):
        hybrid_layer_types(3)


def test_hybrid_has_no_task_specific_position_or_operand_features() -> None:
    model = KDAHybridAdditionModel()
    assert model.position_encoding == "none"
    assert not hasattr(model, "operand_a_projection")
    assert not hasattr(model, "operand_b_projection")
    assert model.layer_types == ("kda", "kda", "kda", "gated_mla")
    # Keep capacity within one percent of the 237,917-parameter deployed baseline.
    assert abs(parameter_count(model) - 237_917) / 237_917 < 0.01


def test_gated_mla_is_causal() -> None:
    mx.random.seed(5001)
    attention = GatedMultiHeadLatentAttention(
        dims=16,
        num_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        head_dim=8,
    )
    x = mx.random.normal((2, 9, 16))
    changed = mx.concatenate((x[:, :5], x[:, 5:] + 0.5), axis=1)
    original_prefix = attention(x)[:, :5]
    changed_prefix = attention(changed)[:, :5]
    mx.eval(original_prefix, changed_prefix)
    np.testing.assert_allclose(
        np.asarray(original_prefix),
        np.asarray(changed_prefix),
        rtol=1e-5,
        atol=1e-6,
    )


def test_complete_hybrid_has_differentiable_training_path() -> None:
    mx.random.seed(5002)
    model = KDAHybridAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        mla_head_dim=8,
    )
    model.train()
    inputs = mx.random.randint(0, 13, shape=(2, 12))
    targets = mx.random.randint(0, 13, shape=(2, 12))
    loss, gradients = nn.value_and_grad(model, masked_addition_loss)(model, inputs, targets)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss))


def test_checkpoint_metadata_forbids_alignment_features(tmp_path) -> None:
    config = HybridTrainConfig(
        schedule=parse_schedule("3:2"),
        output_dir=tmp_path,
    )
    metadata = _architecture_metadata(config)
    assert metadata["position_encoding"] == "none"
    assert metadata["arithmetic_column_sinusoid"] is False
    assert metadata["aligned_operand_context"] is False
    assert metadata["reversed_answer_digits"] is True


def test_adaptive_batch_size_tapers_for_quadratic_mla(tmp_path) -> None:
    config = HybridTrainConfig(
        schedule=parse_schedule("1364:2"),
        output_dir=tmp_path,
        batch_size=4,
        max_batch_size=64,
    )
    assert [_stage_batch_size(config, digits) for digits in (128, 256, 512, 1024, 1364)] == [
        64,
        32,
        16,
        8,
        4,
    ]


def test_training_can_log_before_first_validation(tmp_path) -> None:
    config = HybridTrainConfig(
        schedule=parse_schedule("1:2"),
        output_dir=tmp_path,
        batch_size=1,
        eval_batch_size=1,
        dims=16,
        num_heads=1,
        num_layers=4,
        learning_rate=1e-4,
        lr_decay_steps=2,
        warmup_steps=1,
        log_every=1,
        eval_every=2,
        checkpoint_every=10,
        q_lora_rank=8,
        kv_lora_rank=8,
        mla_head_dim=8,
        adaptive_batch_size=False,
        early_stop_exact_evals=0,
    )
    train(config)

    events = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("train") == 2
    assert [event["event"] for event in events].count("validation") == 1
    assert (tmp_path / "best.json").exists()
