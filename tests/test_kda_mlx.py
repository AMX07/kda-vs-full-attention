import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
from mlx.utils import tree_flatten  # noqa: E402

from karpathygpt_hw.kda_mlx import (  # noqa: E402
    KDAAdditionModel,
    addition_aligned_operand_indices,
    addition_column_position_ids,
    gated_delta_metal_trainable,
    gated_delta_metal_with_final_state,
    gated_delta_reference,
    masked_addition_loss,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="custom KDA training kernel requires Apple Metal",
)


def test_addition_column_position_ids_align_operands_and_shifted_answer() -> None:
    # For three-digit inputs, positions are "AAA+BBB=SSSS". The hidden state at
    # '=' predicts the least-significant answer digit, so it receives column 0.
    assert np.array_equal(
        np.asarray(addition_column_position_ids(12)),
        np.array([2, 1, 0, 4, 2, 1, 0, 0, 1, 2, 3, 4], dtype=np.int32),
    )


def test_addition_column_model_preserves_output_shape() -> None:
    model = KDAAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=1,
        use_fused_kernel=True,
        position_encoding="addition_column",
    )
    logits = model(mx.zeros((2, 12), dtype=mx.int32))
    mx.eval(logits)
    assert logits.shape == (2, 12, 13)


def test_aligned_context_maps_only_answer_queries_to_operand_pairs() -> None:
    operand_a, operand_b, mask = addition_aligned_operand_indices(12)
    assert np.array_equal(np.asarray(operand_a)[7:10], [2, 1, 0])
    assert np.array_equal(np.asarray(operand_b)[7:10], [6, 5, 4])
    assert np.array_equal(
        np.asarray(mask),
        [False, False, False, False, False, False, False, True, True, True, False, False],
    )


def test_addition_column_context_model_preserves_output_shape() -> None:
    model = KDAAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=1,
        use_fused_kernel=True,
        position_encoding="addition_column_context",
    )
    logits = model(mx.zeros((2, 12), dtype=mx.int32))
    mx.eval(logits)
    assert logits.shape == (2, 12, 13)


def test_addition_column_context_does_not_read_future_answer_tokens() -> None:
    model = KDAAdditionModel(
        dims=16,
        num_heads=1,
        num_layers=1,
        use_fused_kernel=True,
        position_encoding="addition_column_context",
    )
    original = np.array([[2, 3, 4, 1, 5, 6, 7, 12, 2, 2, 2, 2]], dtype=np.int32)
    changed_future = original.copy()
    changed_future[:, 8:] = np.array([[8, 9, 10, 11]], dtype=np.int32)
    original_logits = model(mx.array(original))[:, 7, :]
    changed_logits = model(mx.array(changed_future))[:, 7, :]
    mx.eval(original_logits, changed_logits)
    np.testing.assert_allclose(
        np.asarray(original_logits),
        np.asarray(changed_logits),
        rtol=1e-5,
        atol=1e-5,
    )


def _value_and_grad(function, inputs, weights):
    def loss(*arguments):
        return mx.sum(function(*arguments) * weights)

    value, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(*inputs)
    mx.eval(value, gradients)
    return np.asarray(value), [np.asarray(gradient) for gradient in gradients]


def test_prefill_returns_reference_output_and_final_state() -> None:
    rng = np.random.default_rng(1701)
    shape = (2, 7, 2, 4)
    q = mx.array(rng.normal(size=shape).astype(np.float32) * 0.1)
    k = mx.array(rng.normal(size=shape).astype(np.float32) * 0.1)
    v = mx.array(rng.normal(size=shape).astype(np.float32) * 0.1)
    retention = mx.array(rng.uniform(0.8, 1.0, size=shape).astype(np.float32))
    beta = mx.array(rng.uniform(0.0, 1.0, size=shape[:-1]).astype(np.float32))

    output, final_state = gated_delta_metal_with_final_state(q, k, v, retention, beta)
    reference_output = gated_delta_reference(q, k, v, retention, beta)
    zero_query = mx.zeros_like(q[:, -1])
    _, reference_state = gated_delta_metal_with_final_state(
        mx.concatenate((q[:, :-1], zero_query[:, None]), axis=1),
        k,
        v,
        retention,
        beta,
    )
    mx.eval(output, final_state, reference_output, reference_state)
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(reference_output), rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(final_state), np.asarray(reference_state), rtol=1e-6, atol=1e-6
    )


def test_one_step_gradients_match_closed_form() -> None:
    q = mx.array([[[[0.2, -0.3]]]], dtype=mx.float32)
    k = mx.array([[[[0.4, 0.1]]]], dtype=mx.float32)
    v = mx.array([[[[0.7, -0.2]]]], dtype=mx.float32)
    retention = mx.array([[[[0.9, 0.8]]]], dtype=mx.float32)
    beta = mx.array([[[0.5]]], dtype=mx.float32)
    weights = mx.array([[[[0.6, -0.4]]]], dtype=mx.float32)

    value, gradients = _value_and_grad(
        gated_delta_metal_trainable,
        (q, k, v, retention, beta),
        weights,
    )
    expected = [
        np.array([[[[0.1, 0.025]]]], dtype=np.float32),
        np.array([[[[0.05, -0.075]]]], dtype=np.float32),
        np.array([[[[0.015, -0.01]]]], dtype=np.float32),
        np.zeros((1, 1, 1, 2), dtype=np.float32),
        np.array([[[0.025]]], dtype=np.float32),
    ]

    np.testing.assert_allclose(value, 0.0125, rtol=1e-6, atol=1e-7)
    for actual, target in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=1e-6, atol=1e-7)


def test_checkpointed_multichunk_vjp_matches_reference() -> None:
    mx.random.seed(4259)
    shape = (2, 19, 2, 4)  # 19 tokens crosses the 16-token checkpoint boundary.
    inputs = (
        mx.random.normal(shape) * 0.1,
        mx.random.normal(shape) * 0.1,
        mx.random.normal(shape) * 0.1,
        mx.full(shape, 0.97),
        mx.full((2, 19, 2), 0.4),
    )
    weights = mx.random.normal(shape)

    reference_value, reference_gradients = _value_and_grad(
        gated_delta_reference,
        inputs,
        weights,
    )
    fused_value, fused_gradients = _value_and_grad(
        gated_delta_metal_trainable,
        inputs,
        weights,
    )

    np.testing.assert_allclose(fused_value, reference_value, rtol=2e-5, atol=2e-6)
    for fused, reference in zip(fused_gradients, reference_gradients, strict=True):
        # MLX's einsum reference accumulates these reductions slightly
        # differently on the GPU; the fused kernel is exact on the closed-form
        # test above and stays within the expected reduction-rounding envelope.
        np.testing.assert_allclose(fused, reference, rtol=3e-3, atol=5e-5)


def test_fused_recurrence_is_causal() -> None:
    mx.random.seed(4260)
    shape = (1, 19, 2, 4)
    inputs = [
        mx.random.normal(shape) * 0.1,
        mx.random.normal(shape) * 0.1,
        mx.random.normal(shape) * 0.1,
        mx.full(shape, 0.97),
        mx.full((1, 19, 2), 0.4),
    ]
    cutoff = 7
    perturbed = []
    for value in inputs:
        future_noise = mx.random.normal(value.shape) * 0.2
        positions = mx.arange(value.shape[1])[None, :, None]
        if value.ndim == 4:
            positions = positions[..., None]
        perturbed.append(mx.where(positions > cutoff, value + future_noise, value))

    reference = gated_delta_reference(*inputs)
    fused = gated_delta_metal_trainable(*inputs)
    fused_perturbed = gated_delta_metal_trainable(*perturbed)
    mx.eval(reference, fused, fused_perturbed)

    np.testing.assert_allclose(fused, reference, rtol=3e-4, atol=2e-6)
    np.testing.assert_array_equal(
        np.asarray(fused[:, : cutoff + 1]),
        np.asarray(fused_perturbed[:, : cutoff + 1]),
    )

    def prefix_loss(*arguments):
        return mx.sum(gated_delta_metal_trainable(*arguments)[:, : cutoff + 1])

    gradients = mx.grad(prefix_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
    mx.eval(gradients)
    for gradient in gradients:
        np.testing.assert_array_equal(
            np.asarray(gradient[:, cutoff + 1 :]),
            np.zeros_like(np.asarray(gradient[:, cutoff + 1 :])),
        )


def test_fused_path_updates_a_complete_model() -> None:
    mx.random.seed(4261)
    model = KDAAdditionModel(
        vocab_size=13,
        dims=16,
        num_heads=1,
        num_layers=1,
        use_fused_kernel=True,
    )
    model.train()
    optimizer = optim.AdamW(learning_rate=1e-3)
    inputs = mx.random.randint(0, 13, shape=(2, 12))
    targets = mx.random.randint(0, 13, shape=(2, 12))
    targets = mx.where(mx.arange(12)[None] < 7, -1, targets)
    value_and_grad = nn.value_and_grad(model, masked_addition_loss)
    before = dict(tree_flatten(model.trainable_parameters()))

    loss, gradients = value_and_grad(model, inputs, targets)
    optimizer.update(model, gradients)
    after = dict(tree_flatten(model.trainable_parameters()))
    mx.eval(loss, after, optimizer.state)

    assert np.isfinite(float(loss))
    assert any(
        not np.array_equal(np.asarray(before[name]), np.asarray(after[name])) for name in before
    )


def test_long_memory_initialization_spans_requested_half_lives() -> None:
    model = KDAAdditionModel(
        dims=64,
        num_heads=4,
        decay_half_life_range=(8.0, 4_096.0),
    )
    bias = model.blocks[0].attention.decay_bias
    retention = mx.exp(-5.0 * mx.sigmoid(bias))
    half_lives = mx.log(mx.array(0.5)) / mx.log(retention)
    mx.eval(half_lives)
    assert float(mx.min(half_lives)) == pytest.approx(8.0, rel=1e-4)
    assert float(mx.max(half_lives)) == pytest.approx(4_096.0, rel=1e-3)
