from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from karpathygpt_hw.hybrid_addition import KDAHybridAdditionModel  # noqa: E402
from karpathygpt_hw.kda_mlx import masked_addition_loss  # noqa: E402
from karpathygpt_hw.variable_addition import (  # noqa: E402
    IGNORE_INDEX,
    AdditionExample,
    AdditionLayout,
    decode,
    valid_layouts,
)
from karpathygpt_hw.variable_hybrid_addition import (  # noqa: E402
    make_mlx_batch,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="hybrid training uses the custom Metal KDA VJP",
)


def test_mlx_batch_is_variable_unpadded_and_reversed() -> None:
    batch = make_mlx_batch((AdditionExample(7, 23), AdditionExample(8, 34)))
    assert batch.layout == AdditionLayout(1, 2, 2)
    assert batch.inputs.shape == (2, batch.layout.model_input_length)
    assert decode(batch.inputs[0]) == "7+23=03"
    assert np.all(batch.targets[:, : batch.layout.prompt_length - 1] == IGNORE_INDEX)
    assert np.all(batch.targets[:, batch.layout.prompt_length - 1 :] != IGNORE_INDEX)
    assert len({layout.model_input_length for layout in valid_layouts(3)}) > 1


def test_mlx_batch_rejects_tensor_padding_between_layouts() -> None:
    with pytest.raises(ValueError, match="one exact layout"):
        make_mlx_batch((AdditionExample(7, 23), AdditionExample(12, 34)))


def test_complete_hybrid_is_causal() -> None:
    mx.random.seed(7001)
    model = KDAHybridAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        mla_head_dim=8,
    )
    model.eval()
    first = mx.random.randint(0, 13, shape=(1, 12))
    second = mx.concatenate((first[:, :6], (first[:, 6:] + 1) % 13), axis=1)
    logits = model(mx.concatenate((first, second), axis=0))
    mx.eval(logits)
    np.testing.assert_allclose(
        np.asarray(logits[0, :6]),
        np.asarray(logits[1, :6]),
        rtol=2e-5,
        atol=2e-6,
    )


def test_variable_batch_uses_fused_differentiable_training_path() -> None:
    mx.random.seed(7002)
    model = KDAHybridAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        mla_head_dim=8,
    )
    batch = make_mlx_batch((AdditionExample(123, 456), AdditionExample(321, 678)))
    model.train()
    loss, gradients = nn.value_and_grad(model, masked_addition_loss)(
        model,
        mx.array(batch.inputs),
        mx.array(batch.targets),
    )
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss))
